from whisper_streaming.base import OnlineProcessorInterface, ASRBase
import argparse

import sys
import logging
import mlx.core as mx

from simul_whisper.config import AlignAttConfig
from simul_whisper.simul_whisper import PaddedAlignAttWhisper

logger = logging.getLogger(__name__)

def simulwhisper_args(parser):
    group = parser.add_argument_group('Whisper arguments')
    group.add_argument('--model_path', type=str, default='./base.pt', 
                        help='The file path to the Whisper .pt model. If not present on the filesystem, the model is downloaded automatically.')
    group.add_argument('--model_name', type=str, default='small', 
                        help='Model name for alignment heads selection (tiny, base, small, medium, large, large-v1, large-v2, large-v3, etc.)')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.")
    group.add_argument("--decoder",type=str, default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('CoreML acceleration')
    group.add_argument('--use_coreml', action='store_true',
                        help='Use CoreML encoder for faster inference on Apple Silicon (3-5x speedup)')
    group.add_argument('--coreml_encoder_path', type=str, default=None,
                        help='Path to CoreML encoder .mlpackage/.mlmodelc directory (auto-detected if not specified)')
    group.add_argument('--coreml_compute_units', type=str, default='ALL', choices=['ALL', 'CPU_AND_NE', 'CPU_ONLY'],
                        help='CoreML compute units: ALL (default), CPU_AND_NE (Neural Engine), or CPU_ONLY')

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=30.0, 
                        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--audio_min_len', type=float, default=0.0, 
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=25, 
                        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
                            'until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model. ')

    group = parser.add_argument_group('Truncation of the last decoded word (from Simul-Whisper)')
    group.add_argument('--cif_ckpt_path', type=str, default=None, 
                        help='The file path to the Simul-Whisper\'s CIF model checkpoint that detects whether there is' \
                        'end of word at the end of the chunk. If not, the last decoded space-separated word is truncated ' \
                        'because it is often wrong -- transcribing a word in the middle.' \
                        'The CIF model adapted for the Whisper model version should be used. ' \
                        'Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . ' \
                        'Note that there is no model for large-v3.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False, 
                       help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. " \
                       ". If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. " \
                        "Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.")

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=None, help="Max context tokens for the model. Default is 0.")

    group = parser.add_argument_group("VAD configuration")
    group.add_argument('--vad_silence_ms', type=int, default=500, 
                        help='Minimum silence duration in milliseconds before VAD detects end of speech (default: 500ms)')


def simul_asr_factory(args):
    # Convert string log level to logging constant
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    decoder = args.decoder
    if args.beams > 1:
        if decoder == "greedy":
            raise ValueError("Invalid 'greedy' decoder type for beams > 1. Use 'beam'.")
        elif decoder is None or decoder == "beam":
            decoder = "beam"
        else:
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
    else:
        # When beam_size=1, always use greedy decoder for efficiency
        if decoder == "beam":
            logger.warning("BeamSearchDecoder with beam_size=1 is inefficient. Forcing GreedyDecoder instead.")
        decoder = "greedy" 
    
    a = { v:getattr(args, v) for v in ["model_path", "model_name", "cif_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens', "logdir", "vad_silence_ms",
                                       "use_coreml", "coreml_encoder_path", "coreml_compute_units"
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")
    logger.info(f"Arguments: {a}")
    asr = SimulWhisperASR(**a)
    return asr, SimulWhisperOnline(asr)

class SimulWhisperASR(ASRBase):
    
    sep = " "

    def __init__(self, language, model_path, model_name, cif_ckpt_path, frame_threshold, audio_max_len, audio_min_len, segment_length, beams, task,
                 decoder_type, never_fire, init_prompt, static_init_prompt, max_context_tokens, logdir, vad_silence_ms,
                 use_coreml=False, coreml_encoder_path=None, coreml_compute_units="ALL"):
        cfg = AlignAttConfig(
            model_path=model_path,
            model_name=model_name,
            segment_length=segment_length,
            frame_threshold=frame_threshold,
            language=language,
            audio_max_len=audio_max_len,
            audio_min_len=audio_min_len,
            cif_ckpt_path=cif_ckpt_path,
            decoder_type=decoder_type, #"greedy" if beams==1 else "beam",
            beam_size=beams,
            task=task,
            never_fire=never_fire,
            init_prompt=init_prompt,
            max_context_tokens=max_context_tokens,
            static_init_prompt=static_init_prompt,
            logdir=logdir,
            vad_silence_ms=vad_silence_ms,
            # CoreML encoder options
            use_coreml_encoder=use_coreml,
            coreml_encoder_path=coreml_encoder_path,
            coreml_compute_units=coreml_compute_units,
        )
        logger.info(f"Language: {language}")
        self.model = PaddedAlignAttWhisper(cfg)

    def transcribe(self, audio, init_prompt=""):
        logger.info("SimulWhisperASR's transcribe() should not be used. It's here only temporarily." \
        "Instead, use SimulWhisperOnline.process_iter().")
        raise NotImplementedError("Use SimulWhisperOnline.process_iter() instead of transcribe().")

    def warmup(self, audio, init_prompt=""):
        self.model.insert_audio(mx.array(audio))
        self.model.infer(True)
        self.model.refresh_segment(complete=True)
    
    def use_vad(self):
        print("VAD not implemented",file=sys.stderr)

    def set_translate_task(self):
        # this is not used. Translate task is set another way.
        pass


class SimulWhisperOnline(OnlineProcessorInterface):

    def __init__(self, asr):
        self.model = asr.model
        self.file = None
        self.init()

    def init(self, offset=None):
        self.audio_chunks = []
        if offset is not None:
            self.offset = offset
        else:
            self.offset = 0
        self.is_last = False
        self.beg = self.offset
        self.end = self.offset

        self.audio_bufer_offset = self.offset
        self.last_ts = (-1,-1)
        self.model.refresh_segment(complete=True)

        self.unicode_buffer = []  # hide incomplete unicode character for the next iteration

    def insert_audio_chunk(self, audio):
        self.audio_chunks.append(mx.array(audio))

    def timestamped_text(self, tokens, generation):
        if not generation:
            return []

        pr = generation["progress"]
        # The 'result' in generation can be stale if tokens were modified by hide_incomplete_unicode,
        # causing a mismatch. Always re-splitting tokens ensures consistency.
        split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)

        frames = [p["most_attended_frames"][0] for p in pr]
        if frames and self.unicode_buffer != []:
            a = [frames[0]] * len(self.unicode_buffer)
            frames = a + frames
            
        tokens = tokens.copy()
        ret = []
        for sw,st in zip(split_words,split_tokens):
            b = None
            for stt in st:
                t,f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = (b*0.02, e*0.02, sw)
            ret.append(out)
            logger.debug(f"TS-WORD:\t{' '.join(map(str, out))}")
        return ret

    def hide_incomplete_unicode(self, tokens):
        """Sometimes, the last token is an imcomplete unicode character, e.g. a part of "ň" or "ř".
        Without this, the outputs can end with '�' = Unicode Replacement Character, and the next output also
        starts with '�'.
        This function hides the last incomplete unicode character and adds it in the next iteration.
        """
        if self.unicode_buffer != []:
            logger.debug(f"Hiding incomplete unicode character: {self.unicode_buffer}")
            tokens = self.unicode_buffer + tokens
            self.unicode_buffer = []  # clear the buffer after processing
        chars, _ = self.model.tokenizer.split_tokens_on_unicode(tokens)
        if len(chars) > 0 and chars[-1].endswith('�'):
            self.unicode_buffer = tokens[-1:]  # keep the last incomplete unicode character
            logger.debug(f"Hiding incomplete unicode character: {tokens[-1:]}")
            return tokens[:-1]  # remove the last token, which is incomplete unicode character
        return tokens

    def process_iter(self):
        import time
        import logging
        
        # Enable forced evaluation for accurate timing when in DEBUG mode
        FORCE_EVAL = logger.isEnabledFor(logging.DEBUG)
        
        t_start = time.time()
        logger.debug(f"[PERF] SimulWhisperOnline.process_iter() started")
        
        t_audio = time.time()
        if len(self.audio_chunks) == 0:
            audio = None
        else:
            audio = mx.concatenate(self.audio_chunks, axis=0)
            if audio.shape[0] == 0:
                audio = None
            else:
                self.end += audio.shape[0] / self.SAMPLING_RATE
        self.audio_chunks = []
        if FORCE_EVAL and audio is not None:
            mx.eval(audio)
        logger.debug(f"[PERF]   audio chunk processing: {time.time()-t_audio:.4f}s")
        
        t_insert = time.time()
        self.audio_bufer_offset += self.model.insert_audio(audio)
        if FORCE_EVAL:
            # insert_audio doesn't return mx arrays, just offset
            pass
        logger.debug(f"[PERF]   insert_audio: {time.time()-t_insert:.4f}s")
        
        tokens, generation_progress = self.model.infer(is_last=self.is_last)

        t_post = time.time()
        tokens = self.hide_incomplete_unicode(tokens)

        # word-level timestamps
        ts_words = self.timestamped_text(tokens, generation_progress)

        text = self.model.tokenizer.decode(tokens)

        if len(text) == 0:
            logger.debug(f"[PERF]   post-processing: {time.time()-t_post:.4f}s")
            logger.debug(f"[PERF] SimulWhisperOnline.process_iter() total: {time.time()-t_start:.4f}s")
            return (None,None,"")
        self.beg = ts_words[0][0]+self.audio_bufer_offset  # it should be this
        self.beg = max(self.beg, self.last_ts[0]+1)  # but let's create the timestamps non-decreasing -- at least last beg + 1 
        if self.is_last:
            e = self.end
        else:
            e = ts_words[-1][1]+self.audio_bufer_offset
        e = max(e, self.last_ts[1]+1)

        self.last_ts = (self.beg, e)
        
        logger.debug(f"[PERF]   post-processing: {time.time()-t_post:.4f}s")
        logger.debug(f"[PERF] SimulWhisperOnline.process_iter() total: {time.time()-t_start:.4f}s")
        return (self.beg,e,text)

    def finish(self):
        logger.info("Finish")
        self.is_last = True
        o = self.process_iter()
        self.is_last = False
        self.model.refresh_segment(complete=True)
        return o
    

if __name__ == "__main__":
    import os
    from whisper_streaming.whisper_online_main import main_simulation_from_file
    
    # Start MLX memory logging if requested
    memory_logger = None
    if os.environ.get('MLX_MEMORY_LOG'):
        try:
            from mlx_memory_monitor import MemoryLogger
            log_file = os.environ.get('MLX_MEMORY_LOG')
            memory_logger = MemoryLogger(log_file, interval=0.1, console_output=False)
            memory_logger.start()
            print(f"MLX memory logging enabled: {log_file}")
        except ImportError:
            print("Warning: mlx_memory_monitor not found, memory logging disabled")
    
    try:
        main_simulation_from_file(simul_asr_factory, add_args=simulwhisper_args)
    finally:
        if memory_logger:
            memory_logger.stop()