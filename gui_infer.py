import numpy as np
import torch
import torchaudio
import click
import gradio as gr
import tempfile
import gc
import traceback
import os
from slicer import Slicer

from infer import (
    load_models, 
    load_audio, 
    apply_fade, 
    process_segment
)

# Global variables for models
global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, device
svc_model = vocoder = rmvpe = hubert = rms_extractor = spk2idx = dataset_cfg = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_models(model_path):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg
    
    # Default to FP16, but will be overridden in processing
    use_fp16 = True
    
    # Clean up memory before loading models
    if svc_model is not None:
        del svc_model
        del vocoder
        del rmvpe
        del hubert
        del rms_extractor
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # Check if the model file exists
        if not os.path.exists(model_path):
            return [], f"❌ 错误: 找不到模型文件: {model_path}"
        
        svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model_path, device, use_fp16)
        available_speakers = list(spk2idx.keys())
        return available_speakers, f"✅ 模型加载成功！可用说话人: {', '.join(available_speakers)}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return [], f"❌ 加载模型出错: {str(e)}\n\n详细信息: {error_trace}"

def process_with_progress(
    progress=gr.Progress(),
    input_audio=None,
    speaker=None,
    key_shift=0,
    infer_steps=32,
    robust_f0=1,
    use_fp16=True,
    # Advanced CFG parameters
    ds_cfg_strength=0.1,
    spk_cfg_strength=1.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=6,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    # Slicer parameters
    slicer_threshold=-30.0,
    slicer_min_length=3000,
    slicer_min_interval=100,
    slicer_hop_size=10,
    slicer_max_sil_kept=200
):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg
    
    # Fixed parameters
    target_loudness = -18.0
    restore_loudness = True
    fade_duration = 20.0
    sliced_inference = False
    
    # Input validation
    if input_audio is None:
        return None, "❌ 错误: 未提供输入音频。"
    
    if svc_model is None:
        return None, "❌ 错误: 模型未加载。请先加载模型。"
    
    if speaker is None or speaker not in spk2idx:
        return None, f"❌ 错误: 无效的说话人选择。可用说话人: {', '.join(spk2idx.keys())}"
    
    # Process the audio
    try:
        # Update status message
        progress(0, desc="处理中: 加载音频...")
        
        # Convert speaker name to ID
        speaker_id = spk2idx[speaker]
        
        # Get config from loaded model
        hop_length = 512
        sample_rate = 44100
        
        # Handle negative skip_layers value as None
        if cfg_skip_layers < 0:
            cfg_skip_layers_value = None
        else:
            cfg_skip_layers_value = cfg_skip_layers
        
        # Load audio
        audio = load_audio(input_audio, sample_rate)
        
        # Initialize Slicer
        slicer = Slicer(
            sr=sample_rate,
            threshold=slicer_threshold,
            min_length=slicer_min_length,
            min_interval=slicer_min_interval,
            hop_size=slicer_hop_size,
            max_sil_kept=slicer_max_sil_kept
        )
        
        progress(0.1, desc="处理中: 切分音频...")
        # Slice the input audio
        segments_with_pos = slicer.slice(audio)
        
        if not segments_with_pos:
            return None, "❌ 错误: 在输入文件中未找到有效的音频片段。"
        
        # Calculate fade size in samples
        fade_samples = int(fade_duration * sample_rate / 1000)
        
        # Process segments
        result_audio = np.zeros(len(audio) + fade_samples)  # Extra space for potential overlap
        
        progress(0.2, desc="处理中: 开始转换...")
        
        with torch.no_grad():
            for i, (start_sample, chunk) in enumerate(segments_with_pos):
                segment_progress = 0.2 + (0.7 * (i / len(segments_with_pos)))
                progress(segment_progress, desc=f"处理中: 片段 {i+1}/{len(segments_with_pos)}")
                
                # Process the segment
                audio_out = process_segment(
                    chunk, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                    speaker_id, sample_rate, hop_length, device,
                    key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                    skip_cfg_strength, cfg_skip_layers_value, cfg_rescale,
                    cvec_downsample_rate, target_loudness, restore_loudness, sliced_inference,
                    robust_f0, use_fp16
                )
                
                # Ensure consistent length
                expected_length = len(chunk)
                if len(audio_out) > expected_length:
                    audio_out = audio_out[:expected_length]
                elif len(audio_out) < expected_length:
                    audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
                
                # Apply fades
                if i > 0:  # Not first segment
                    audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
                    result_audio[start_sample:start_sample + fade_samples] *= \
                        np.linspace(1, 0, fade_samples)  # Fade out previous
                
                if i < len(segments_with_pos) - 1:  # Not last segment
                    audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)  # Fade out
                
                # Add to result
                result_audio[start_sample:start_sample + len(audio_out)] += audio_out
                
                # Clean up memory after each segment
                torch.cuda.empty_cache()
        
        progress(0.9, desc="处理中: 完成音频...")
        # Trim any extra padding
        result_audio = result_audio[:len(audio)]
        
        # Create a temporary file to save the result
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
        
        # Save output
        torchaudio.save(output_path, torch.from_numpy(result_audio).unsqueeze(0).float(), sample_rate)
        
        progress(1.0, desc="处理完成!")
        return (sample_rate, result_audio), f"✅ 转换完成! 已转换为 **{speaker}** 并调整 **{key_shift}** 个半音。"
        
    except RuntimeError as e:
        # Handle CUDA out of memory errors
        if "CUDA out of memory" in str(e):
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            return None, f"❌ 错误: 内存不足。请尝试更短的音频文件或减少推理步骤。"
        else:
            return None, f"❌ 转换过程中出错: {str(e)}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, f"❌ 转换过程中出错: {str(e)}\n\n详细信息: {error_trace}"
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

def create_ui():
    # CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .footer {
        margin-top: 20px;
        text-align: center;
        font-size: 0.9em;
        color: #666;
    }
    .title {
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 20px;
        color: #666;
    }
    .button-primary {
        background-color: #5460DE !important;
    }
    .output-message {
        margin-top: 10px;
        padding: 10px;
        border-radius: 4px;
        background-color: #f8f9fa;
        border-left: 4px solid #5460DE;
    }
    .error-message {
        color: #d62828;
        font-weight: bold;
    }
    .success-message {
        color: #588157;
        font-weight: bold;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #5460DE;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    """
    
    # No need to initialize models automatically - let user provide their model path first
    available_speakers = []
    init_message = "⏳ 请加载模型以开始使用。"
    
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="RIFT-SVC 声音转换") as app:
        gr.HTML("""
        <div class="title">
            <h1>🎤 RIFT-SVC 歌声音色转换</h1>
        </div>
        <div class="subtitle">
            <h3>使用 RIFT-SVC 模型将歌声或语音转换为目标音色</h3>
        </div>
        <div class="info-box">
            <p>🔗 <strong>想要微调自己的说话人？</strong> 请访问 <a href="https://github.com/Pur1zumu/RIFT-SVC" target="_blank">RIFT-SVC GitHub 仓库</a> 获取完整的训练和微调指南。</p>
        </div>
        <div class="info-box">
            <p>📝 <strong>注意：</strong> 为获得最佳效果，请使用背景噪音较少的干净音频。</p>
        </div>
        """)
        
        with gr.Row():
            # Left column (input parameters)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📥 输入")
                    model_path = gr.Textbox(label="模型路径", value="", placeholder="请输入您的模型文件路径", interactive=True)
                    reload_btn = gr.Button("🔄 加载模型", elem_id="reload_btn")
                    input_audio = gr.Audio(label="输入音频文件", type="filepath", elem_id="input_audio")
                
                with gr.Accordion("⚙️ 基本参数", open=True):
                    speaker = gr.Dropdown(label="目标说话人", interactive=True, elem_id="speaker")
                    key_shift = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label="音调调整（半音）", elem_id="key_shift")
                    infer_steps = gr.Slider(minimum=8, maximum=64, step=1, value=32, label="推理步数", elem_id="infer_steps", 
                                           info="更低的值 = 更快但质量较低，更高的值 = 更慢但质量更好")
                    use_fp16 = gr.Checkbox(label="使用 FP16 精度", value=True, info="启用以提高性能，在某些GPU上可能会降低精度", elem_id="use_fp16")
                    robust_f0 = gr.Radio(choices=[0, 1, 2], value=1, label="音高滤波", 
                                        info="0=无，1=轻度过滤，2=强力过滤（有助于解决断音/破音问题）", 
                                        elem_id="robust_f0")
                
                with gr.Accordion("🔬 高级CFG参数", open=True):
                    ds_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, 
                                               label="内容向量引导强度", 
                                               info="更高的值可以改善内容保留和咬字清晰度。过高会用力过猛。", 
                                               elem_id="ds_cfg_strength")
                    spk_cfg_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=1.0, 
                                                label="说话人引导强度", 
                                                info="更高的值可以增强说话人相似度。过高可能导致音色失真。", 
                                                elem_id="spk_cfg_strength")
                    skip_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, 
                                                 label="层引导强度（实验性功能）", 
                                                 info="增强指定层的特征渲染。效果取决于目标层的功能。", 
                                                 elem_id="skip_cfg_strength")
                    cfg_skip_layers = gr.Number(value=-1, label="CFG跳过层（实验性功能）", precision=0, 
                                               info="目标增强层下标，-1为禁用此功能", 
                                               elem_id="cfg_skip_layers")
                    cfg_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, 
                                           label="CFG重缩放因子", 
                                           info="约束整体引导强度。当引导效果过于强烈时使用调高该值。", 
                                           elem_id="cfg_rescale")
                    cvec_downsample_rate = gr.Radio(choices=[1, 2, 4, 8], value=2, 
                                                  label="用于反向引导的内容向量下采样率", 
                                                  info="更高的值（可能）可以提高内容清晰度。", 
                                                  elem_id="cvec_downsample_rate")
                
                with gr.Accordion("✂️ 切片参数", open=False):
                    slicer_threshold = gr.Slider(minimum=-60.0, maximum=-20.0, step=0.1, value=-30.0, 
                                                label="阈值 (dB)", 
                                                info="静音检测阈值", 
                                                elem_id="slicer_threshold")
                    slicer_min_length = gr.Slider(minimum=1000, maximum=10000, step=100, value=3000, 
                                                 label="最小长度 (毫秒)", 
                                                 info="最小片段长度", 
                                                 elem_id="slicer_min_length")
                    slicer_min_interval = gr.Slider(minimum=10, maximum=500, step=10, value=100, 
                                                   label="最小静音间隔 (毫秒)", 
                                                   info="分割片段的最小间隔", 
                                                   elem_id="slicer_min_interval")
                    slicer_hop_size = gr.Slider(minimum=1, maximum=20, step=1, value=10, 
                                              label="跳跃大小 (毫秒)", 
                                              info="片段检测窗口大小", 
                                              elem_id="slicer_hop_size")
                    slicer_max_sil_kept = gr.Slider(minimum=10, maximum=1000, step=10, value=200, 
                                                  label="保留的最大静音 (毫秒)", 
                                                  info="保留在每个片段边缘的最大静音长度", 
                                                  elem_id="slicer_max_sil_kept")
            
            # Right column (output)
            with gr.Column(scale=1):
                convert_btn = gr.Button("🎵 转换声音", variant="primary", elem_id="convert_btn")
                gr.Markdown("### 📤 输出")
                output_audio = gr.Audio(label="转换后的音频", elem_id="output_audio", autoplay=False, show_share_button=False)
                output_message = gr.Markdown(init_message, elem_id="output_message", elem_classes="output-message")
                
                gr.HTML("""
                <div class="info-box">
                    <h4>🔍 快速提示</h4>
                    <ul>
                        <li><strong>音调调整：</strong> 以半音为单位上调或下调音高。</li>
                        <li><strong>推理步骤：</strong> 步骤越多 = 质量越好但速度越慢。</li>
                        <li><strong>音高滤波：</strong> 有助于提高具有挑战性的音频中的音高稳定性。</li>
                        <li><strong>CFG参数：</strong> 调整转换质量和音色。</li>
                    </ul>
                </div>
                """)
        
        # Define button click events
        def load_model_and_update_speakers(model_path):
            # Call initialize_models to load the model
            available_speakers, message = initialize_models(model_path)
            
            # Explicitly update the dropdown with new speakers
            if available_speakers and len(available_speakers) > 0:
                return gr.update(choices=available_speakers, value=available_speakers[0]), message
            else:
                return gr.update(choices=[], value=None), message
        
        reload_btn.click(
            fn=load_model_and_update_speakers,
            inputs=[model_path],
            outputs=[speaker, output_message]
        )
        
        # Updated convert button click event
        convert_btn.click(
            fn=lambda: "⏳ 处理中... 请稍候。",
            inputs=None,
            outputs=output_message,
            queue=False
        ).then(
            fn=process_with_progress,
            inputs=[
                input_audio, speaker, key_shift, infer_steps, robust_f0, use_fp16,
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength, cfg_skip_layers, cfg_rescale, cvec_downsample_rate,
                slicer_threshold, slicer_min_length, slicer_min_interval, slicer_hop_size, slicer_max_sil_kept
            ],
            outputs=[output_audio, output_message],
            show_progress_on=output_audio
        )
    
    return app

@click.command()
@click.option('--share', is_flag=True, help='Share the app')
def main(share=False):
    app = create_ui()
    app.launch(share=share)

if __name__ == "__main__":
    main()