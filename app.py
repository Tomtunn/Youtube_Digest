from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from tqdm.auto import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu" 
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize(tokens, model, max_summary_length=512, device="cpu"):
    """
    tokens: input token (tensor of token)
    max_summary_length: maximum number of tokens in summary (int)
    device: device to run model (str)
    
    return summarize input text using model (str)
    """
    token_input = tokens.to(device)
    model = model.to(device)
    summary_ids = model.generate(token_input, min_length=int(max_summary_length//5), max_length=max_summary_length, num_beams=4,  early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def split_token_chunks(text, tokenizer, max_tokens=1024, overlap=0.2, device="cpu"):
    """
    text: input text (str)
    max_tokens: maximum number of tokens per chunk (int)
    overlap: number of overlapping tokens between chunks (int)
    device: device to run model (str)
    
    create overlapping token chunks from input
    
    return: list of token chunks (list of tensor)
    """
    tokens = tokenizer("summarize: " + text, return_tensors="pt")["input_ids"].to(device)
    token_chunks = [tokens[:, i:i+max_tokens] for i in range(0, tokens.shape[1], max_tokens-int(max_tokens*overlap))] # split token into chunks
    return token_chunks
    
def summarize_long_text(text, max_summary_length=512, level=0, max_token_length=1024, device="cpu"):
    """
    text: input text (str)
    max_summary_length: maximum number of tokens in summary (int)
    level: level of recursion (int)
    max_token_length: maximum number of tokens per chunk (int)
    device: device to run model (str)

    recursively summarize long text by splitting into chunks
    
    return summarize input text using model (str)
    """
    level = level + 1
    print(f"Level: {level}")
    token_chunks = split_token_chunks(text, tokenizer, device=device)
    summary_ls = []
    for token_chunk in tqdm(token_chunks):
        summary = summarize(token_chunk, model, max_summary_length=int(max_summary_length//3), device=device)
        summary_ls.append(summary)
    summary_concat = " ".join(summary_ls)
    tokens_summary_concat = tokenizer(summary_concat, return_tensors="pt")["input_ids"]
    
    if tokens_summary_concat.shape[1] > max_token_length:
        return summarize_long_text(summary_concat, max_summary_length=max_summary_length, level=level,  device=device)
    if level > 100:
        print("Level > 100, return summary_concat")
        return summary_concat
    else:
        final_summary = summarize(tokens_summary_concat, model, max_summary_length=max_summary_length, device=device)
        return final_summary

def get_transcript(video_link):
    """
    video_link: youtube video link (str)
    
    return transcript of the video (str)
    """
    video_id = video_link.split("v=")[1]
    eng_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([line['text'] for line in eng_transcript])
    return transcript


def main():
    print(device)
    st.title('Video Summarizer')
    link = st.text_input('Enter the youtube video link')
    max_summary_length = st.slider('Select the maximum length of the summary', 100, 5000, 512, step=100)

    if st.button('Generate Summary'):
        if link:
            try:
                status_text = st.empty()
                progress_bar = st.progress(25)
                
                status_text.text('Loading the transcript...')
                progress_bar.progress(50)
                transcript = get_transcript(link)
                print('transcript:', 'pass')
                
            except Exception as e:
                st.write('Error:', e)
                st.write('Please enter a valid youtube video link')
                
            try:
                status_text.text('Summarizing the transcript...')
                progress_bar.progress(75)
                summary = summarize_long_text(transcript, max_summary_length=max_summary_length, device=device)
                print('summary:', 'pass')
                
                status_text.text('Finish summarized!!!')
                progress_bar.progress(100)
                st.markdown(summary)
                
            except Exception as e:
                st.write('Error:', e)
                st.write('Cannot summarize the video return transcript instead')
                st.write(transcript)    
                 
        else:
            st.write('Please enter a youtube video link') 

if __name__ == "__main__":
    main()