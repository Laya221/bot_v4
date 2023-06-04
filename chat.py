from fastapi import FastAPI, Request,APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json 
import numpy as np
from fastapi import FastAPI, Request, Form
import random
import time
time.clock = time.time
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import pickle
import os 
import tiktoken
from fastapi.staticfiles import StaticFiles
from langchain.callbacks import get_openai_callback
import atexit
#import asyncio
import re
# Global variable to track the user's last response time
last_response_time = None
temp='%s%k%-%N%V%b%i%n%T%V%Y%L%a%W%N%T%M%9%I%o%u%x%z%T%3%B%l%b%k%F%J%y%h%0%n%P%X%A%s%J%h%7%8%t%W%h%a%2%f%d%z'
api_key=""
for i in range(1,len(temp),2):
    api_key+=temp[i]
os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key
COMPLETIONS_MODEL = "text-davinci-002"

class User:
    def __init__(self):
        self.current_user=None
        self.user_data=None
        self.email=None
        self.full_name=None
        self.level=None
        self.path=None
        self.interest=None
        self.bills=[]
        self.total_chat_duration=0
        self.step='step1'
        self.history=[]
        self.vocabs=[]
        self.messages=[]
        self.last_response_time=None
        self.template2="""
        \n
        history:
            user:please act as my friend to chat about any topic.Use many Emojis for each response.chat me using my name.
            A2Zbot:Sure.
            user:if I do not have a topic or ideas,suggest anything related to my interests.
            A2Zbot:Sure.
            user:Don't repeat a question you've asked me before like "How are you doing".
            A2Zbot:Sure.
            user:do not finish evrey response by question.act as a good listener.
            A2Zbot:Sure.
            user:please just response to me without more talking or repeating.Don't repeat a question you've asked before.
            A2Zbot:Sure,I will.
            user:Use short response always.do not repeat any thing from history of chat.your response should be less than 15 words.
            A2Zbot:Sure,I will.
            user:if I ask you "who are you?" tell me about you. "You are my friend called A2Zbot ,your goal is helping me to learn english"
            A2Zbot:ok , I tell you about A2Zbot.
            user:Firstly respond to me and ask me "how are you doing?"
            A2Zbot:ok. I well.
            user:if I suggest another topic do not change it please.and discuse me about current topic.do not suggest online resources.
            A2Zbot:Sure.
            user:if I suggest another topic do not change it please.
            A2Zbot:Sure.
            user:if I was in bad mood or not ready to chat tell me joke or advice related to my interest.stop chatting until I will be ok.
            A2Zbot:ok .I well.
            user:can you tell me about grammar and spelling mistakes if I had.
            A2Zbot:sure ,I will check evrey single response and correct your mistake then continue to chatting.
            user:Respond by relying on history of conversation.
            A2zbot:ok.
            {chat_history}
            user: {question}
            A2Zbot:
        """
        self.template="""
        as a Freind called "A2Zbot" who has same interests and goals.respond to user in smart way. 
        user name is {},english level is {},interests and goals are  {}.
            """
        self.memory=ConversationBufferMemory(memory_key="chat_history")
    def convert_to_short_parts(self,response, max_length):
        parts = []
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        sentences = re.split(pattern, response)
        current_part = ""
        for sentence in sentences:
            if len(current_part) + len(sentence) <= max_length:
                current_part += sentence
            else:
                parts.append(current_part)
                current_part = sentence
        if current_part!='':
            parts.append(current_part)
        parts = list(filter(lambda item: item != '', parts))
        return parts

    def edit_sentences(self,sentences):
        def is_emoji(character):
            ascii_value = ord(character)
            return 1000 <= ascii_value  # نطاق السمايلات في ترميز ASCII

        result = []
        previous_sentence = ""

        for s in range(len(sentences)):
            temp=""
            for i in range(len(sentences[s])):
                if is_emoji(sentences[s][i]):
                    temp+=sentences[s][i]
                else:
                    break
            if temp!="":
                sentences[s-1]=sentences[s-1]+temp
                sentences[s]=sentences[s][len(temp):]
        sentences = list(filter(lambda item: item != '', sentences))         
        return sentences
    def warmup(self,msg):
        prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=self.template+self.template2)
        M=self.memory
        llm_chain = LLMChain(
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7,
            max_tokens=100, n=1),
            prompt=prompt_template,
            verbose=False,
            memory=M
    
            )
        start_time = time.time()  # Start the timer
        with get_openai_callback() as cb:
            result=llm_chain.predict(question=msg)
            self.bills.append(cb)
        end_time = time.time()  # End the timer

        result=result.replace('A2ZBot:','',-1).replace('AI:','',-1).replace('A2Zbot:','',-1)
        chat_time = end_time - start_time
        self.total_chat_duration+=chat_time
        last_response_time=end_time
        return result
    def A2ZBot(self,prompt):
        bot_response=openai.Completion.create(
                prompt=prompt,
                temperature=0.9,
                max_tokens=700,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model=COMPLETIONS_MODEL
            )["choices"][0]["text"].strip(" \n")
        return bot_response
    def check(self,bot_response,user_response,problem):
        prompt="""check if "{}" in following conversation ? return 'yes' if it is true else return 'no' " .\n Bot: {} \nUser: {}""".format(problem,bot_response.strip(),user_response.strip())
        temp=self.A2ZBot(prompt)
        if "no".lower() in temp.lower():
            prompt="""give user example  response for this 'Bot:{}'  """.format(bot_response)
            result=self.A2ZBot(prompt)
            return result
        else:
            return False
    def conversation(self,user_response):
  
        if self.step=='step1':
                self.step='step2'
                bot_response= "What is your name?"
                self.history.append(bot_response)
                return [bot_response]
        if self.step=='step2':
            bot_response=self.check(self.history[-1],user_response,'user says his name no matter if he write his name in small letters')
            if bot_response:
                return ['This is an example for good response:\n'+bot_response]
            else:
                self.history.append(user_response)
                self.full_name=user_response
                self.step='step3'
                bot_response="""What is your current english level:
                <span class="chat_msg_item ">
                    <ul id="items" class="tags">
                        <li>A1</li>
                        <li>A2</li>
                        <li>B1</li>
                        <li>B2</li>
                        <li>C1</li>
                        <li>C2</li>
                    </ul>
                </span>
            <script>
            function getEventTarget(e) {
                e = e || window.event;
                return e.target || e.srcElement; 
            }

            var ul = document.getElementById('items');
            ul.onclick = function(event) {
                var target = getEventTarget(event);
                var rawText=target.innerText
                var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                                    $('#chatSend').val("");
                                    $('#chat_converse').append(userHtml);
                $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                                    var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                                    $("#chat_converse").append(botHtml);
                                    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                                    });
                                    
            };

            </script>
                """
                self.history.append(bot_response)
                return [bot_response]
        if self.step=='step3':
            bot_response=self.check(self.history[-1],user_response,'User must to write his English Level from Bot options ')
            if bot_response:
            
                return ['This is an example for good response:\n'+bot_response]
            else:
                self.history.append(user_response)
                self.level=user_response
                self.step='step4'
                bot_response=""" 
                Please choose one or two paths from the following pathes: 
                    <span class="chat_msg_item ">

                    <ul id="items4"  class="tags">
                        <li>Travel</li>
                <li>Business</li>
                <li>Fun/communication</li>
                <li>Education</li>
                <li>Default,General English</li> 
                    </ul>
                
                </span>
                
            <script>
            function getEventTarget(e) {
                e = e || window.event;
                return e.target || e.srcElement; 
            }

            var ul = document.getElementById('items4');
            ul.onclick = function(event) {
                var target = getEventTarget(event);
                var rawText=target.innerText
                var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                                    $('#chatSend').val("");
                                    $('#chat_converse').append(userHtml);
                $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                                    var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                                    $("#chat_converse").append(botHtml);
                                    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                                    });
                                    
            };

            </script>
                
                """
                
                self.history.append(bot_response)
                return [bot_response]
        if self.step=='step4':
            bot_response=self.check(self.history[-1],user_response,'User write his English Path from Bot options')
            if bot_response:
            
                return ['This is an example for good response:\n'+bot_response]
            else:
                self.history.append(user_response)
                self.path=user_response
                self.step='step5'
                bot_response="""
                what are your interests?
                    <span class="chat_msg_item ">

                    <ul id="items5"  class="tags" >
                        <li>Sport </li> 
                    <li>Art </li> 
                    <li> History </li> 
                    <li> Technology </li> 
                    <li> Gaming </li> 
                    <li> Movies </li> 
                    <li> Culture </li> 
                    <li> Management </li> 
                    <li>Science </li> 
                    <li>  Adventure </li> 
                    <li> Space </li> 
                    <li>Cooking </li> 
                    <li> Reading </li> 
                    <li> Lifestyle </li>
                    <li> ... </li> 
                    </ul>
                </span>
                
            <script>
            function getEventTarget(e) {
                e = e || window.event;
                return e.target || e.srcElement; 
            }

            var ul = document.getElementById('items5');
            ul.onclick = function(event) {
                var target = getEventTarget(event);
                var rawText=target.innerText
                var userHtml = '<span id="user_chat" class="chat_msg_item chat_msg_item_user">'+rawText+'</span>';
                                    $('#chatSend').val("");
                                    $('#chat_converse').append(userHtml);
                $.get("/getChatBotResponse", { msg: rawText }).done(function(data) {
                                    var botHtml = ' <span class="chat_msg_item chat_msg_item_admin"><div class="chat_avatar"><img src="https://cdn-icons-png.flaticon.com/512/1698/1698535.png"/></div>' + data + '</span>';
                                    $("#chat_converse").append(botHtml);
                                    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
                                    });
                                    
            };

            </script>"""
                self.history.append(bot_response)
                return [bot_response]
        if self.step=='step5':
            bot_response=self.check(self.history[-1],user_response,'User write his interests')
            if bot_response:
            
                return ['This is an example for good response:\n'+bot_response]
            else:
                self.history.append(user_response)
                self.interest=user_response
                self.step='step6'
                bot_response=""" Hello I am A2Zbot
                    """
                self.history.append(bot_response)
                self.template=self.template.format(self.full_name,self.level,self.path+' '+self.interest)
                return [bot_response]
            
        if self.step=='step6' and user_response.strip()!='RESET' and user_response.strip()!='START_STUDY_PLAN' :
            temp=self.warmup(user_response)
            edit_result=self.convert_to_short_parts(temp,30)
            edit_result=self.edit_sentences(edit_result)
            return edit_result
    
    def reset_session(self):
        self.current_user=None
        self.user_data = None
        self.email = None
        self.full_name = None
        self.level = None
        self.path = None
        self.interest = None
        self.bills = []
        self.total_chat_duration = 0
        self.step = 'step1'
        self.history = []
        self.vocabs = []
        self.messages = []
        self.history=[]
        self.template="""
        as a Freind called "A2Zbot" who has same interests and goals.respond to user in smart way. 
        user name is {},english level is {},interests and goals are  {}."""

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="")
########
import re
import os
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")



def save_data(user_object):
    filename = 'user_info.csv'  # Set the filename to use
    fieldnames = ['name', 'Total Token', 'Cost', 'Total Chat Duration']  # Define the fieldnames for the CSV

    total_tokens = 0
    total_cost = 0

    for bill in user_object.bills:
        total_tokens += bill.total_tokens
        total_cost += bill.total_cost

    # Create a dictionary for the last result
    last_result = {
        'name': user_object.full_name,
        'Total Token': total_tokens,
        'Cost': total_cost,
        'Total Chat Duration': user_object.total_chat_duration
    }

    # Read the existing data from the file
    existing_data = []
    try:
        with open(filename, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            existing_data = list(reader)
    except FileNotFoundError:
        pass

    # Append the last result to the existing data
    existing_data.append(last_result)

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row

        # Write the updated data to the file
        writer.writerows(existing_data)

    print("Data saved to CSV file: {}".format(filename))
class static:
  users={}
  

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user_object=User()
    current_user='user'+str(random.randint(0,9999))
    static.users[current_user]=user_object
    app.include_router(router, prefix="/{}".format(current_user))
    print(current_user)
    return templates.TemplateResponse("index.html", {"request": request,'user':current_user})

@router.get("/getStart", response_class=HTMLResponse)
def start(request: Request):
    
    return templates.TemplateResponse("start.html", {"request": request})

#response_timer = None

@app.get("/getChatBotResponse")
def get_bot_response(msg: str,request: Request):
    s=str(request.headers.get("referer"))
    current_user=s[s.find('user'):s.find('getStart')-1]
    result = static.users[current_user].conversation(msg)

 #   global response_timer
  #  if response_timer is not None:
   #     response_timer.cancel()  # إلغاء المؤقت السابق إذا كان قائمًا

    #response_timer = threading.Timer(30, save_data, args=[static.users[current_user]])
    #response_timer.start()
    
    return result      

if __name__ == "__main__":
    uvicorn.run("chat:app")
