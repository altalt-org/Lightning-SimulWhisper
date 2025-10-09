#!/usr/bin/env python3
"""
Comprehensive visualization of timing performance comparison between CoreML, MLX, and PyTorch implementations.
Creates beautiful Plotly visualizations with outlier filtering for stable analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Color scheme for consistent visualization
COLORS = {
    'coreml': '#FF6B6B',      # Red
    'mlx': '#4ECDC4',         # Teal  
    'pytorch': '#45B7D1',     # Blue
    'whisper_encoder': '#96CEB4'  # Green
}

def load_timing_data():
    """Load timing data from all three implementations"""
    data = {}
    
    implementations = ['coreml', 'mlx', 'pytorch']
    
    for impl in implementations:
        impl_data = {}
        timing_dir = Path(f"timing_logs_{impl}")
        
        if not timing_dir.exists():
            print(f"Warning: {timing_dir} not found, skipping {impl}")
            continue
            
        # Load encoding times
        encoding_file = timing_dir / "encoding_times.csv"
        if encoding_file.exists():
            df_enc = pd.read_csv(encoding_file)
            impl_data['encoding'] = df_enc
            
        # Load decoding times  
        decoding_file = timing_dir / "decoding_times.csv"
        if decoding_file.exists():
            df_dec = pd.read_csv(decoding_file)
            impl_data['decoding'] = df_dec
            
        # Load inference times
        infer_file = timing_dir / "infer_times.csv"
        if infer_file.exists():
            df_infer = pd.read_csv(infer_file)
            impl_data['inference'] = df_infer
            
        data[impl] = impl_data
        
    return data

def filter_outliers(data, column='duration_ms', percentile_range=(10, 90)):
    """Filter outliers by removing top and bottom percentiles"""
    if column not in data.columns:
        return data
        
    lower_bound = data[column].quantile(percentile_range[0] / 100)
    upper_bound = data[column].quantile(percentile_range[1] / 100)
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    print(f"Filtered {len(data) - len(filtered_data)} outliers ({len(data)} -> {len(filtered_data)} rows)")
    return filtered_data

def create_encoding_comparison(data):
    """Create encoding time comparison visualizations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Encoding Time Distribution', 
            'Encoding Time Box Plot',
            'Encoding Time Over Time',
            'Encoding Performance Summary'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Prepare data for comparison
    all_encoding_data = []
    for impl, impl_data in data.items():
        if 'encoding' in impl_data:
            df = impl_data['encoding'].copy()
            df = filter_outliers(df, 'duration_ms')
            df['implementation'] = impl
            all_encoding_data.append(df)
    
    if not all_encoding_data:
        print("No encoding data available")
        return None
        
    combined_df = pd.concat(all_encoding_data, ignore_index=True)
    
    # 1. Distribution comparison
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['duration_ms']
        fig.add_trace(
            go.Histogram(
                x=impl_data,
                name=f'{impl.upper()} Distribution',
                opacity=0.7,
                nbinsx=30,
                marker_color=COLORS.get(impl, '#888888')
            ),
            row=1, col=1
        )
    
    # 2. Box plot comparison
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['duration_ms']
        fig.add_trace(
            go.Box(
                y=impl_data,
                name=f'{impl.upper()}',
                marker_color=COLORS.get(impl, '#888888'),
                boxpoints='outliers'
            ),
            row=1, col=2
        )
    
    # 3. Time series
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl].sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=impl_data['timestamp'],
                y=impl_data['duration_ms'],
                mode='markers',
                name=f'{impl.upper()} Over Time',
                marker_color=COLORS.get(impl, '#888888'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # 4. Performance summary
    summary_stats = []
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['duration_ms']
        stats = {
            'Implementation': impl.upper(),
            'Mean (ms)': impl_data.mean(),
            'Median (ms)': impl_data.median(),
            'Std (ms)': impl_data.std(),
            'Min (ms)': impl_data.min(),
            'Max (ms)': impl_data.max()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create bar chart for mean times
    fig.add_trace(
        go.Bar(
            x=summary_df['Implementation'],
            y=summary_df['Mean (ms)'],
            name='Mean Encoding Time',
            marker_color=[COLORS.get(impl.lower(), '#888888') for impl in summary_df['Implementation']],
            text=[f'{val:.1f}ms' for val in summary_df['Mean (ms)']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Encoding Performance Comparison',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Duration (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Implementation", row=2, col=2)
    fig.update_yaxes(title_text="Mean Duration (ms)", row=2, col=2)
    
    return fig

def create_decoding_comparison(data):
    """Create decoding time comparison visualizations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Decoding Time vs Step Number',
            'Decoding Time Distribution',
            'Total Tokens vs Decoding Time',
            'Decoding Performance by Implementation'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Prepare data for comparison
    all_decoding_data = []
    for impl, impl_data in data.items():
        if 'decoding' in impl_data:
            df = impl_data['decoding'].copy()
            df = filter_outliers(df, 'duration_ms')
            df['implementation'] = impl
            all_decoding_data.append(df)
    
    if not all_decoding_data:
        print("No decoding data available")
        return None
        
    combined_df = pd.concat(all_decoding_data, ignore_index=True)
    
    # 1. Decoding time vs step number
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]
        fig.add_trace(
            go.Scatter(
                x=impl_data['step'],
                y=impl_data['duration_ms'],
                mode='markers',
                name=f'{impl.upper()}',
                marker_color=COLORS.get(impl, '#888888'),
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # 2. Distribution comparison
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['duration_ms']
        fig.add_trace(
            go.Histogram(
                x=impl_data,
                name=f'{impl.upper()} Distribution',
                opacity=0.7,
                nbinsx=30,
                marker_color=COLORS.get(impl, '#888888')
            ),
            row=1, col=2
        )
    
    # 3. Total tokens vs decoding time
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]
        fig.add_trace(
            go.Scatter(
                x=impl_data['total_tokens'],
                y=impl_data['duration_ms'],
                mode='markers',
                name=f'{impl.upper()}',
                marker_color=COLORS.get(impl, '#888888'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # 4. Performance summary
    summary_stats = []
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['duration_ms']
        stats = {
            'Implementation': impl.upper(),
            'Mean (ms)': impl_data.mean(),
            'Median (ms)': impl_data.median(),
            'Std (ms)': impl_data.std()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    fig.add_trace(
        go.Bar(
            x=summary_df['Implementation'],
            y=summary_df['Mean (ms)'],
            name='Mean Decoding Time',
            marker_color=[COLORS.get(impl.lower(), '#888888') for impl in summary_df['Implementation']],
            text=[f'{val:.1f}ms' for val in summary_df['Mean (ms)']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Decoding Performance Comparison',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Step Number", row=1, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Total Tokens", row=2, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Implementation", row=2, col=2)
    fig.update_yaxes(title_text="Mean Duration (ms)", row=2, col=2)
    
    return fig

def create_inference_comparison(data):
    """Create inference time comparison visualizations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Total Inference Time Distribution',
            'Inference Time vs Decode Steps',
            'Inference Performance Over Time',
            'Performance Summary'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Prepare data for comparison
    all_inference_data = []
    for impl, impl_data in data.items():
        if 'inference' in impl_data:
            df = impl_data['inference'].copy()
            df = filter_outliers(df, 'total_duration_ms')
            df['implementation'] = impl
            all_inference_data.append(df)
    
    if not all_inference_data:
        print("No inference data available")
        return None
        
    combined_df = pd.concat(all_inference_data, ignore_index=True)
    
    # 1. Distribution comparison
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['total_duration_ms']
        fig.add_trace(
            go.Histogram(
                x=impl_data,
                name=f'{impl.upper()} Distribution',
                opacity=0.7,
                nbinsx=30,
                marker_color=COLORS.get(impl, '#888888')
            ),
            row=1, col=1
        )
    
    # 2. Inference time vs decode steps
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]
        fig.add_trace(
            go.Scatter(
                x=impl_data['num_decode_steps'],
                y=impl_data['total_duration_ms'],
                mode='markers',
                name=f'{impl.upper()}',
                marker_color=COLORS.get(impl, '#888888'),
                opacity=0.6
            ),
            row=1, col=2
        )
    
    # 3. Time series
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl].sort_values('timestamp')
        fig.add_trace(
            go.Scatter(
                x=impl_data['timestamp'],
                y=impl_data['total_duration_ms'],
                mode='markers',
                name=f'{impl.upper()} Over Time',
                marker_color=COLORS.get(impl, '#888888'),
                opacity=0.6
            ),
            row=2, col=1
        )
    
    # 4. Performance summary
    summary_stats = []
    for impl in combined_df['implementation'].unique():
        impl_data = combined_df[combined_df['implementation'] == impl]['total_duration_ms']
        stats = {
            'Implementation': impl.upper(),
            'Mean (ms)': impl_data.mean(),
            'Median (ms)': impl_data.median(),
            'Std (ms)': impl_data.std(),
            'Min (ms)': impl_data.min(),
            'Max (ms)': impl_data.max()
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    fig.add_trace(
        go.Bar(
            x=summary_df['Implementation'],
            y=summary_df['Mean (ms)'],
            name='Mean Inference Time',
            marker_color=[COLORS.get(impl.lower(), '#888888') for impl in summary_df['Implementation']],
            text=[f'{val:.1f}ms' for val in summary_df['Mean (ms)']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Inference Performance Comparison',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Total Duration (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Number of Decode Steps", row=1, col=2)
    fig.update_yaxes(title_text="Total Duration (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Total Duration (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Implementation", row=2, col=2)
    fig.update_yaxes(title_text="Mean Duration (ms)", row=2, col=2)
    
    return fig

def create_comprehensive_summary(data):
    """Create a comprehensive performance summary"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Encoding Performance',
            'Decoding Performance', 
            'Inference Performance'
        ]
    )
    
    # Collect all performance metrics
    performance_data = {
        'Implementation': [],
        'Encoding_Mean': [],
        'Decoding_Mean': [],
        'Inference_Mean': [],
        'Encoding_Std': [],
        'Decoding_Std': [],
        'Inference_Std': []
    }
    
    for impl, impl_data in data.items():
        performance_data['Implementation'].append(impl.upper())
        
        # Encoding performance
        if 'encoding' in impl_data:
            enc_data = filter_outliers(impl_data['encoding'], 'duration_ms')
            performance_data['Encoding_Mean'].append(enc_data['duration_ms'].mean())
            performance_data['Encoding_Std'].append(enc_data['duration_ms'].std())
        else:
            performance_data['Encoding_Mean'].append(0)
            performance_data['Encoding_Std'].append(0)
        
        # Decoding performance
        if 'decoding' in impl_data:
            dec_data = filter_outliers(impl_data['decoding'], 'duration_ms')
            performance_data['Decoding_Mean'].append(dec_data['duration_ms'].mean())
            performance_data['Decoding_Std'].append(dec_data['duration_ms'].std())
        else:
            performance_data['Decoding_Mean'].append(0)
            performance_data['Decoding_Std'].append(0)
        
        # Inference performance
        if 'inference' in impl_data:
            inf_data = filter_outliers(impl_data['inference'], 'total_duration_ms')
            performance_data['Inference_Mean'].append(inf_data['total_duration_ms'].mean())
            performance_data['Inference_Std'].append(inf_data['total_duration_ms'].std())
        else:
            performance_data['Inference_Mean'].append(0)
            performance_data['Inference_Std'].append(0)
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create bar charts with error bars
    implementations = perf_df['Implementation']
    colors = [COLORS.get(impl.lower(), '#888888') for impl in implementations]
    
    # Encoding
    fig.add_trace(
        go.Bar(
            x=implementations,
            y=perf_df['Encoding_Mean'],
            name='Encoding',
            marker_color=colors,
            error_y=dict(type='data', array=perf_df['Encoding_Std']),
            text=[f'{val:.1f}ms' for val in perf_df['Encoding_Mean']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Decoding
    fig.add_trace(
        go.Bar(
            x=implementations,
            y=perf_df['Decoding_Mean'],
            name='Decoding',
            marker_color=colors,
            error_y=dict(type='data', array=perf_df['Decoding_Std']),
            text=[f'{val:.1f}ms' for val in perf_df['Decoding_Mean']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Inference
    fig.add_trace(
        go.Bar(
            x=implementations,
            y=perf_df['Inference_Mean'],
            name='Inference',
            marker_color=colors,
            error_y=dict(type='data', array=perf_df['Inference_Std']),
            text=[f'{val:.1f}ms' for val in perf_df['Inference_Mean']],
            textposition='auto'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Comprehensive Performance Summary',
        height=500,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Duration (ms)", row=1, col=3)
    
    return fig

def create_scaling_analysis(data):
    """Create scaling analysis visualizations"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Decoding Time vs Sequence Length',
            'Inference Time vs Complexity'
        ]
    )
    
    # Decoding scaling analysis
    for impl, impl_data in data.items():
        if 'decoding' in impl_data:
            df = impl_data['decoding'].copy()
            df = filter_outliers(df, 'duration_ms')
            
            # Group by total_tokens to see scaling
            grouped = df.groupby('total_tokens')['duration_ms'].agg(['mean', 'std', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 3]  # Only include groups with enough samples
            
            fig.add_trace(
                go.Scatter(
                    x=grouped['total_tokens'],
                    y=grouped['mean'],
                    error_y=dict(type='data', array=grouped['std']),
                    mode='markers+lines',
                    name=f'{impl.upper()} Decoding',
                    marker_color=COLORS.get(impl, '#888888')
                ),
                row=1, col=1
            )
    
    # Inference scaling analysis
    for impl, impl_data in data.items():
        if 'inference' in impl_data:
            df = impl_data['inference'].copy()
            df = filter_outliers(df, 'total_duration_ms')
            
            # Group by num_decode_steps to see scaling
            grouped = df.groupby('num_decode_steps')['total_duration_ms'].agg(['mean', 'std', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 3]  # Only include groups with enough samples
            
            fig.add_trace(
                go.Scatter(
                    x=grouped['num_decode_steps'],
                    y=grouped['mean'],
                    error_y=dict(type='data', array=grouped['std']),
                    mode='markers+lines',
                    name=f'{impl.upper()} Inference',
                    marker_color=COLORS.get(impl, '#888888')
                ),
                row=1, col=2
            )
    
    fig.update_layout(
        title='Performance Scaling Analysis',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Total Tokens", row=1, col=1)
    fig.update_yaxes(title_text="Decoding Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Decode Steps", row=1, col=2)
    fig.update_yaxes(title_text="Inference Time (ms)", row=1, col=2)
    
    return fig

def main():
    """Main function to create all visualizations"""
    print("Loading timing data...")
    data = load_timing_data()
    
    if not data:
        print("No timing data found!")
        return
    
    print(f"Loaded data for implementations: {list(data.keys())}")
    
    # Create output directory
    output_dir = Path("timing_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    visualizations = [
        ("encoding_comparison", create_encoding_comparison),
        ("decoding_comparison", create_decoding_comparison),
        ("inference_comparison", create_inference_comparison),
        ("comprehensive_summary", create_comprehensive_summary),
        ("scaling_analysis", create_scaling_analysis)
    ]
    
    for name, create_func in visualizations:
        print(f"Creating {name}...")
        fig = create_func(data)
        
        if fig is not None:
            # Save as HTML
            html_file = output_dir / f"{name}.html"
            fig.write_html(str(html_file))
            
            # Save as PNG
            png_file = output_dir / f"{name}.png"
            fig.write_image(str(png_file), width=1200, height=800)
            
            print(f"  Saved: {html_file} and {png_file}")
        else:
            print(f"  Skipped {name} (no data)")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Open the HTML files in a web browser to view interactive charts!")

if __name__ == "__main__":
    main()
