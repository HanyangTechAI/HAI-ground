"""
## App: NLP App with Streamlit (NLPiffy)
Author: [Seongjin Lee(GirinMan)](https://github.com/GirinMan))\n
Source: [Github](https://github.com/GirinMan/HAI-DialectTranslator/)
Credits: 2022-Fall HAI Team 1 project

실행 방법: streamlit run app.py


"""
# Core Pkgs
import streamlit as st 
import requests
import json

SPAWNER_API = "API LINK"
headers = {'content-type': 'application/json'}

st.set_page_config(layout="wide")

def main():
    """ NLP Based App with Streamlit """
    
    # Title
    st.title("HAI-ground")
    st.markdown("https://github.com/HanyangTechAI/HAI-ground")
    st.markdown("""
        #### Description
        - 입력된 문장에 이어지는 다음 텍스트를 자동으로 생성해 줍니다.
        - GPT-3 모델에 대한 자세한 설명은 [링크](https://openai.com/blog/gpt-3-apps)를 참고하세요.
        - Powered by [KoGPT](https://huggingface.co/kakaobrain/kogpt)(KakaoBrain)
        """)
    
    with st.sidebar:
        st.title("Advanced options")
        max_new_tokens = st.number_input('Max new tokens', min_value=1, max_value=2048, value=32)
        do_sample = st.checkbox('Use sampling', value=True)
        if do_sample:
            num_output = st.number_input('Number of outputs', min_value=1, max_value=5, value=1)
            top_p = st.slider('Top P', min_value=0., max_value=1., value=0.8)
            top_k = st.slider('Top K', min_value=0, value=0) 
            temperature = st.slider('Temperature', min_value=0., max_value=2., value=0.19)
        else:
             num_output = 1
             top_p = 0
             top_k = 0.
             temperature = 0.


    input_area = st.empty()
    input_area.text_area(f"텍스트 입력", "여기에 입력", key='input_txt' ,height=200)

    area = st.container()

    if st.button("Submit"):

        area.subheader("문장 생성 결과")
        input_txt = st.session_state.input_txt

        with area:
            with st.spinner("Generating next sentences..."):
                body = json.dumps({
                    "text": input_txt,
                    "num_sequences":num_output,
                    "max_new_tokens":max_new_tokens,
                    "do_sample":do_sample,
                    "top_k":top_k,
                    "top_p":top_p,
                    "temperature":temperature,
                })
                response = requests.post(url=SPAWNER_API+'generate', headers=headers, data=body).json()
                new_sentences = response["generated"]

        for i, sent in enumerate(new_sentences):
            area.text_area(f'Result {i+1}', sent)

if __name__ == '__main__':
	main()