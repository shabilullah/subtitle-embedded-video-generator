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
    subprocess.call(
        ["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
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

block = gr.Interface(
    fn=translate,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Video(label="Output Video"),
    title="Add Text/Caption to your video - MultiLingual",
    description="Add Text/Caption to your YouTube Shorts with Gradio and OpenAI's Whisper."
)

block.launch()
