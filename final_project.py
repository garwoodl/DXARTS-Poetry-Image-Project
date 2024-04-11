import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
import math
import random

API_KEY = "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"
VIDEO_RES = 100



def poem_to_narrated_video(poem_name, frames_per_transition=VIDEO_RES, n=0):
    """
    Given the name of a poem, run the whole pipeline to create a narrated video
    This function will create new images and new audio files each run
    If n is specified then n photos will be produced.
    """
    # create the video
    print("Text to speech...")
    read_a_poem(poem_filename=poem_name)
    print(f"Imagining poem {poem_name}...")
    imagine_a_poem(poem_filename=poem_name, n=n)
    print("Creating Video...")
    photos_to_video(poem_name=poem_name, frames_per_transition=frames_per_transition)
    combine_audio_video(poem_name=poem_name)
    print("Done.")

    # clean up old files
    unnarrated_file = f"C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_videos\\{poem_name}_video.avi"
    reading_file = f"C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_readings\\{poem_name}.mp3"
    os.remove(unnarrated_file)
    os.remove(reading_file)



def photos_to_video(poem_name, frames_per_transition=VIDEO_RES, length_of_video=5):
    """
    Given a folder of images, this function will create a video that linearly interpolates the pictures into a video
    and then saves it to poem_videos as {poem_name}_unnarrated.avi
        frames_per_transition: the smoothness of the photot blend
        length_of_video: gives the approximate length of the video in seconds (will adjust the fps accordingly)
    """
    video_name = f"{poem_name}_video.avi"
    image_folder = f"C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_images\\{poem_name}_images"

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # Create a black frame
    height, width, _ = cv2.imread(os.path.join(image_folder, images[0])).shape
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Save the black frame as a temporary PNG file
    _, temp_black_frame_path = tempfile.mkstemp(suffix='.png')
    cv2.imwrite(temp_black_frame_path, black_frame)

    # Insert the black frame at the beginning
    images.insert(0, temp_black_frame_path)

    num_images = len(images)
    desired_fps = num_images * frames_per_transition // length_of_video
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_videos")
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), desired_fps, (width, height))

        # Iterate through all images
    for i in range(len(images) - 1):
        img1 = cv2.imread(os.path.join(image_folder, images[i]))
        img2 = cv2.imread(os.path.join(image_folder, images[i + 1]))

        # Generate intermediate frames
        for j in range(frames_per_transition + 1):
            alpha = j / frames_per_transition
            # the way blend_images is set up we need 2 then 1
            blended_frame = blend_images(img2, img1, alpha)
            video.write(blended_frame)

    # Write the last frame (the last image)
    video.write(cv2.imread(os.path.join(image_folder, images[-1])))

    cv2.destroyAllWindows()
    video.release()


def combine_audio_video(poem_name):
    """
    Once the .mp3 file is in poem_readings and .avi is in poem_videos then this function will replace the .avi with
    a video that has narration
    """

    # Load video and audio files
    audio_file = f"C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_readings\\{poem_name}.mp3"
    audio_clip = AudioFileClip(audio_file)

    duration = audio_clip.duration
    # need to get the duration here to make the video the same length
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_videos")
    photos_to_video(poem_name=poem_name, length_of_video=duration)
    video_clip = VideoFileClip(f"{poem_name}_video.avi")

    # Set video clip's audio to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the result to a new video file
    video_clip.write_videofile(f"narrated_{poem_name}.mp4", codec="libx264", audio_codec="aac")

    # Close the clips
    video_clip.close()
    audio_clip.close()


def text_to_speech(text, output_filename):
    '''
    Given a block of text, this function returns a mp3 file called
    '''
    client = OpenAI(api_key=API_KEY)
    if output_filename[-4:] != ".mp3":
        output_filename += '.mp3'

    response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=text
    )

    response.stream_to_file(output_filename)


def read_a_poem(poem_filename):
    """
    Given the name of a single poem in the poetry_collection folder,
    create an mp3 file of a recording of the poem
    """
    if poem_filename[-4:] != '.txt':
        poem_filename += ".txt"

    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poetry_collection")
    poem_file = open(poem_filename, mode='r')
    poem = poem_file.read()
    poem_title = poem_filename[:-4]
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_readings")
    text_to_speech(poem, poem_title)


def blend_images(img1, img2, alpha):
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def imagine_a_poem(poem_filename, n=0, min_stanzas=2):
    """
    Given the name of a single poem in the poetry_collection folder,
    create a folder of photos where each stanza corresponds to an image
    If n is specified then it will only produce n images
    """
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poetry_collection")
    if poem_filename[-4:] != '.txt':
        poem_filename += ".txt"
    poem_file = open(poem_filename, mode='r')
    poem = poem_file.read()

    if n == 0:
        stanza_list = poem.split("\n\n")
        if len(stanza_list) < min_stanzas: # if the poem has too few stanzas then each line is a picture
            stanza_list = poem.split("\n")
    else:
        stanza_list = []
        letters_in_pic = len(poem) // n
        for i in range(n):
            stanza_list.append(poem[i * letters_in_pic: (i + 1) * letters_in_pic])
    
    # Create a directory for the poem if it doesn't exist
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poem_images")
    poem_dir = f'{poem_filename.split(".")[0]}_images'
    os.makedirs(poem_dir, exist_ok=True)

    for i, stanza in enumerate(stanza_list):
        try:
            stanza_image_url = prompt_to_image(stanza)
            image_filename = f"{i}_{poem_filename.split('.')[0]}.png"
            save_image_from_url(stanza_image_url, os.path.join(poem_dir, image_filename))
        except: # want to make it so it only handles safety warnings
            ...
            # error_prompt = """openai.BadRequestError: Error code:
            # 400, content_policy_violation, message, Your request
            # was rejected as a result of our safety system. Your
            # prompt may contain text that is not allowed by our
            # safety system. invalid_request_error"""
            # stanza_image_url = prompt_to_image(error_prompt)
            # image_filename = f"{i}_safety_system_{poem_filename.split('.')[0]}.png"
            # save_image_from_url(stanza_image_url, os.path.join(poem_dir, image_filename))


def prompt_to_image(prompt="dog at a casino"):
    """
    Connects to the OpenAI API and generates a single image based on the prompt
    COSTS MONEY TO RUN
        Use a basic model and low quality while testing code in order to save costs
    """
    client = OpenAI(api_key=API_KEY)
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="standard",  # "hd"
    )
    image_url = response.data[0].url
    return image_url


def save_image_from_url(url, filename):
    """
    Given the url for a generated image, this function saves the photo as filename
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.save(filename)


def get_length_of_poem(poem_filename):
    """
    Return the number of characters in a poem specified by title
    """
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poetry_collection")
    if poem_filename[-4:] != '.txt':
        poem_filename += ".txt"
    poem_file = open(poem_filename, mode='r')
    poem = poem_file.read()

    return len(poem)


def main():
    os.chdir("C:\\Users\\logan\\OneDrive\\School_Files\\DXARTS 480\\FinalProject\\poetry_collection")
    filenames = os.listdir()

    for file in filenames:
        number_of_characters_per_pic = random.randint(60, 80)
        poem_title = file.split('.')[0]
        poem_length = get_length_of_poem(poem_title)
        n = math.ceil(poem_length / number_of_characters_per_pic)
        print(f"making {poem_title}...")
        poem_to_narrated_video(poem_title, n=n)


if __name__ == "__main__":
    main()