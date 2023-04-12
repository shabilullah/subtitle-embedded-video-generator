import gradio as gr 
import os
import subprocess
import whisper

model = whisper.load_model("medium")

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

from whisper.utils import get_writer

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"

def translate(input_video):
    audio_file = video2mp3(input_video)
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio_file,**translate_options)
    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    writer = get_writer("vtt", output_dir)
    writer(result, audio_filename)
    subtitle = os.path.join(output_dir, audio_filename + ".vtt").replace("\\", "/")
    output_video = os.path.join(output_dir, audio_filename + "_subtitled.mp4").replace("\\", "/")
    os.system(f"ffmpeg -y -i {input_video} -vf subtitles={subtitle} {output_video}")
    return output_video

title = "Add Text/Caption to your YouTube Shorts - MultiLingual"

block = gr.Blocks()

with block:
    with gr.Group():
        with gr.Box(): 
            with gr.Row().style():
                inp_video = gr.Video(
                    label="Input Video",
                    type="filepath",
                    mirror_webcam = False
                )
                op_video = gr.Video()
            btn = gr.Button("Generate Subtitle Video")
        btn.click(translate, inputs=[inp_video], outputs=[op_video])
        gr.HTML('''
        <div class="footer">
            <p>Model by <a href="https://github.com/openai/whisper" style="text-decoration: underline;" target="_blank">OpenAI</a> - Gradio App by <a href="https://twitter.com/1littlecoder" style="text-decoration: underline;" target="_blank">1littlecoder</a>
            </p>
        </div>
        ''')

block.launch(debug = True)
