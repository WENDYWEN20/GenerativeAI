import streamlit as st
import os
import openai
import sys
import time
import requests
import pytube
from pytube import YouTube
from zipfile import ZipFile
from hugchat import hugchat
from hugchat.login import Login

st.set_page_config(page_title = "🤗💬 HugChat")
with st.sidebar:
    st.image("https://1.bp.blogspot.com/-qQryqABhdhA/XcC3lJupTKI/AAAAAAAAAzA/MOYu3P_DFRsmNkpjD9j813_SOugPgoBLACLcBGAsYHQ/s1600/h1.png")
    st.header('LLM makes your life easier and more interesting', divider='rainbow')
    choice=st.radio("Navigation",["🤗💬 HugChat","OpenAI","Speech-To-Text","Text-To-Image"])


if choice=="🤗💬 HugChat":
    st.header('This application allows Hugging Face LLM to answer your questions for free.', divider='rainbow')
    st.info('Creat your acccount with Hugging Face following the link, then enter your Email and Login credential. https://huggingface.co/welcome')

    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        st.success('HuggingFace Login credentials already provided!', icon='✅')
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your Hugging Face credentials', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
   # st.markdown('📖!')

# Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "📖!How may I help you?"}]
##Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Function for generating LLM response
    def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
        sign = Login(email, passwd)
        cookies = sign.login()
    # Create ChatBot                        
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

# User-provided prompt
    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

# Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, hf_email, hf_pass) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
    
    
if choice=="OpenAI":
    st.header('This application allows u to see differences between HuggingFace and ChatGPT answers.', divider='rainbow')
    # with open('hidden.txt') as file:
    #     openai.api_key=file.read()
    if 'OpenAIapi_key' in st.secrets:
        st.success('OpenAI API key already provided!', icon='✅')
        openai.api_key = st.secrets['OpenAIapi_key']

    else:
        openai.api_key = st.text_input('Enter OpenAI API key:', type='password')
        
        if not openai.api_key:
            st.warning('Please enter your OpenAI API key!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
        
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})        

if choice=="Speech-To-Text":
    st.header('This application allows u to convert your youtube speech to text.', divider='rainbow')  
    st.markdown('Progress Bar')
    bar = st.progress(0)
    # custom functions
    # retrieving audio files from youtube
            # Sidebar
    #st.header('Youtube URL')

    if 'AssemblyAI_key' in st.secrets:
        st.success('AssemblyAI API key already provided!', icon='✅')
        api_key = st.secrets['AssemblyAI_key']

    else:
        api_key = st.text_input('Enter AssemblyAI API key:', type='password')
        
        if not api_key:
            st.warning('Please enter your AssemblyAI API key', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    
    URL = st.text_input('Enter Youtube video URL:')
    st.text('Example Youtube URL: https://www.youtube.com/watch?v=FAyKDaXEAgc')
    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Go')
        # Run custom functions if URL is entered 
    
    def get_yt(URL):
        yt = YouTube(URL)
        stream = yt.streams.get_audio_only()
        stream.download() 
         #st.info('2. Audio file has been retrieved from YouTube video')
        bar.progress(10)
        
# 3. Upload YouTube audio file to AssemblyAI
    def transcribe_yt():
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            if file.endswith(".mp4"):
                mp4_file = os.path.join(current_dir, file)
                #print(mp4_file)
        filename = mp4_file
        bar.progress(20)
        
        def read_file(filename, chunk_size=5242880):
            with open(filename, 'rb') as _file:
                while True:
                    data = _file.read(chunk_size)
                    if not data:
                        break
                    yield data
        headers = {'authorization': api_key}
        response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))
        audio_url = response.json()['upload_url']
        #st.info('3. YouTube audio file has been uploaded to AssemblyAI')
        bar.progress(30)

        # 4. Transcribe uploaded audio file
        endpoint = "https://api.assemblyai.com/v2/transcript"

        json = {
        "audio_url": audio_url
        }

        headers = {
            "authorization": api_key,
            "content-type": "application/json"
        }

        transcript_input_response = requests.post(endpoint, json=json, headers=headers)

        #st.info('4. Transcribing uploaded file')
        bar.progress(40)
          
        # 5. Extract transcript ID
        transcript_id = transcript_input_response.json()["id"]
        #st.info('5. Extract transcript ID')
        bar.progress(50)

        # 6. Retrieve transcription results
        endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        headers = {
            "authorization": api_key,
        }
        transcript_output_response = requests.get(endpoint, headers=headers)
        #st.info('6. Retrieve transcription results')
        bar.progress(60)

        # Check if transcription is complete
        from time import sleep

        while transcript_output_response.json()['status'] != 'completed':
            sleep(5)
            st.warning('Transcription is processing ...')
            transcript_output_response = requests.get(endpoint, headers=headers)
        
        bar.progress(100)

        # 7. Print transcribed text
        st.header('Output')
        st.success(transcript_output_response.json()["text"])

        # 8. Save transcribed text to file

        # Save as TXT file
        yt_txt = open('yt.txt', 'w')
        yt_txt.write(transcript_output_response.json()["text"])
        yt_txt.close()

        # Save as SRT file
        srt_endpoint = endpoint + "/srt"
        srt_response = requests.get(srt_endpoint, headers=headers)
        with open("yt.srt", "w") as _file:
            _file.write(srt_response.text)
        
        zip_file = ZipFile('transcription.zip', 'w')
        zip_file.write('yt.txt')
        zip_file.write('yt.srt')
        zip_file.close()       
    
    if submit_button:
        get_yt(URL)
        transcribe_yt()

        with open("transcription.zip", "rb") as zip_download:
            btn = st.download_button(
                label="Download ZIP",
                data=zip_download,
                file_name="transcription.zip",
                mime="application/zip"
                )
        
if choice=="Text-To-Image":
    pass
#     st.header('This application allows Hugging Face to convert Text to Image.', divider='rainbow')
#     st.info('Creat your acccount with Hugging Face following the link, then enter your Email and Login credential. https://huggingface.co/welcome')

#     if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
#         st.success('HuggingFace Login credentials already provided!', icon='✅')
#         hf_email = st.secrets['EMAIL']
#         hf_pass = st.secrets['PASS']
#     else:
#         hf_email = st.text_input('Enter E-mail:', type='password')
#         hf_pass = st.text_input('Enter password:', type='password')
#         if not (hf_email and hf_pass):
#             st.warning('Please enter your Hugging Face credentials', icon='⚠️')
#         else:
#             st.success('Proceed to entering your Text-To-Image message!', icon='👉')
    
#     import torch
#     from diffusers import StableDiffusionPipeline
#     pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
#     pipe = pipe.to("cuda")
#     promt=st.text_input("What imgae u would like to make?")
#     image=pipe(prompt).image[0]
#     image.save("image.png")
    
    
