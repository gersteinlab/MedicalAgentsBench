import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
try:
    import google.generativeai as genai
except:
    pass
#from openai import OpenAI
from openai import AzureOpenAI, OpenAI
from pptree import *
from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_ENDPOINT"),
                api_key = os.getenv("AZURE_API_KEY"),
                api_version = os.getenv("AZURE_API_VERSION"),
            )
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        elif self.model_info in ['o1-mini', 'o3-mini']:
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_ENDPOINT_2"),
                api_key = os.getenv("AZURE_API_KEY_2"),
                api_version = os.getenv("AZURE_API_VERSION_2"),
            )
            self.messages = [
                {"role": "user", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        elif self.model_info == 'deepseek-V3':
            self.client = OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
            )
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
            

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        
        elif self.model_info in ['o1-mini', 'o3-mini']:
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=self.messages
            )
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        
        elif self.model_info == 'deepseek-V3':
            response = self.client.chat.completions.create(
                model='deepseek-ai/DeepSeek-V3',
                messages=self.messages
            )
            return response.choices[0].message.content, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        else:
            raise ValueError(f"Unsupported model: {self.model_info}")

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                response = self.client.chat.completions.create(
                    model=self.model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        
        elif self.model_info in ['o1-mini', 'o3-mini']:
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model_info,
                messages=self.messages,
            )
            return {'0': response.choices[0].message.content}, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        elif self.model_info == 'deepseek-V3':
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            responses = {}
            for temperature in temperatures:
                response = self.client.chat.completions.create(
                    model='deepseek-ai/DeepSeek-V3',
                    messages=self.messages,
                    temperature=temperature,
                )
                responses[temperature] = response.choices[0].message.content
                
            return responses, {'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens}
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses, {'prompt_tokens': 0, 'completion_tokens': 0}

class Group:
    def __init__(self, goal, members, question, examplers=None, model='gpt-4o-mini'):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery, delivery_usage = lead_member.chat(delivery_prompt)
            except:
                delivery, delivery_usage = assist_members[0].chat(delivery_prompt)

            total_usage['prompt_tokens'] += delivery_usage['prompt_tokens']
            total_usage['completion_tokens'] += delivery_usage['completion_tokens']

            investigations = []
            for a_mem in assist_members:
                investigation, investigation_usage = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
                total_usage['prompt_tokens'] += investigation_usage['prompt_tokens']
                total_usage['completion_tokens'] += investigation_usage['completion_tokens']
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response, response_usage = lead_member.chat(investigation_prompt)
            total_usage['prompt_tokens'] += response_usage['prompt_tokens']
            total_usage['completion_tokens'] += response_usage['completion_tokens']

            return response, response_usage

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai, None
    elif 'gpt' in model_name:
        client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_ENDPOINT"),
                api_key = os.getenv("AZURE_API_KEY"),
                api_version = os.getenv("AZURE_API_VERSION"),
            )
        return None, client
    elif 'o1' in model_name or 'o3' in model_name:
        client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_ENDPOINT_2"),
                api_key = os.getenv("AZURE_API_KEY_2"),
                api_version = os.getenv("AZURE_API_VERSION_2"),
            )
        return None, client
    elif 'deepseek' in model_name:
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
        )
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset_dir, split):
    test_qa = []
    examplers = []

    test_path = f'{dataset_dir}/{split}.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'{dataset_dir}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset): 
    question = sample['question'] + " Options: "
    options = []
    for k, v in sample['options'].items():
        options.append("({}) {}".format(k, v))
    random.shuffle(options)
    question += " ".join(options)
    return question, None

def determine_difficulty(question, difficulty, model):
    if difficulty != 'adaptive':
        return difficulty, {'prompt_tokens': 0, 'completion_tokens': 0}
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model)
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response, response_usage = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic', response_usage
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate', response_usage
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced', response_usage

def process_basic_query(question, examplers, model, args):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    new_examplers = []
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    random.shuffle(examplers)
    for ie, exampler in enumerate(examplers[:5]):
        tmp_exampler = {}
        exampler_question = exampler['question']
        choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
        random.shuffle(choices)
        exampler_question += " " + ' '.join(choices)
        exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
        exampler_reason, exampler_reason_usage = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

        total_usage['prompt_tokens'] += exampler_reason_usage['prompt_tokens']
        total_usage['completion_tokens'] += exampler_reason_usage['completion_tokens']
        tmp_exampler['question'] = exampler_question
        tmp_exampler['reason'] = exampler_reason
        tmp_exampler['answer'] = exampler_answer
        new_examplers.append(tmp_exampler)

    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model)
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision, final_decision_usage = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)

    total_usage['prompt_tokens'] += final_decision_usage['prompt_tokens']
    total_usage['completion_tokens'] += final_decision_usage['completion_tokens']

    decision_agent = Agent(instruction='You are an answer parser.', role='Answer Parser', model_info='gpt-4o-mini')
    decision_agent.chat('You are an answer parser.')
    decision_answer, decision_answer_usage = decision_agent.chat(f'The following are multiple choice questions (with answers) about medical knowledge.\n\nHere is the question: {question}\n\nOnly output the single letter from the following {final_decision}.', img_path=None)

    total_usage['prompt_tokens'] += decision_answer_usage['prompt_tokens']
    total_usage['completion_tokens'] += decision_answer_usage['completion_tokens']

    return {
        'majority': final_decision,
        'answer': decision_answer
    }, total_usage

def process_intermediate_query(question, examplers, model, args):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 5  # You can adjust this number as needed
    recruited, recruited_usage = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")
    total_usage['prompt_tokens'] += recruited_usage['prompt_tokens']
    total_usage['completion_tokens'] += recruited_usage['completion_tokens']

    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model) 
    random.shuffle(examplers)
    for ie, exampler in enumerate(examplers[:5]):
        exampler_question = f"[Example {ie+1}]\n" + exampler['question']
        options = [f"({k}) {v}" for k, v in exampler['options'].items()]
        random.shuffle(options)
        exampler_question += " " + " ".join(options)
        exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
        exampler_reason, exampler_reason_usage = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
        total_usage['prompt_tokens'] += exampler_reason_usage['prompt_tokens']
        total_usage['completion_tokens'] += exampler_reason_usage['completion_tokens']
        
        exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
        fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    for k, v in agent_dict.items():
        opinion, opinion_usage = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        total_usage['prompt_tokens'] += opinion_usage['prompt_tokens']
        total_usage['completion_tokens'] += opinion_usage['completion_tokens']
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        report, report_usage = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        total_usage['prompt_tokens'] += report_usage['prompt_tokens']
        total_usage['completion_tokens'] += report_usage['completion_tokens']
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate, participate_usage = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                total_usage['prompt_tokens'] += participate_usage['prompt_tokens']
                total_usage['completion_tokens'] += participate_usage['completion_tokens']
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert, chosen_expert_usage = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    total_usage['prompt_tokens'] += chosen_expert_usage['prompt_tokens']
                    total_usage['completion_tokens'] += chosen_expert_usage['completion_tokens']
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question, specific_question_usage = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        total_usage['prompt_tokens'] += specific_question_usage['prompt_tokens']
                        total_usage['completion_tokens'] += specific_question_usage['completion_tokens']
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        # Collect the current answers even when discussion stops early.
        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response, response_usage = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            total_usage['prompt_tokens'] += response_usage['prompt_tokens']
            total_usage['completion_tokens'] += response_usage['completion_tokens']
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

        if num_yes == 0:
            break
        if n < num_rounds:
            round_opinions[n + 1] = tmp_final_answer.copy()

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    _decision, _decision_usage = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    total_usage['prompt_tokens'] += _decision_usage['prompt_tokens']
    total_usage['completion_tokens'] += _decision_usage['completion_tokens']
    final_decision = {'majority': _decision}

    emoji = '\U0001F468\u200D\u2696\uFE0F'
    print(f"{emoji} moderator's final decision (by majority vote):", _decision)
    print()

    # Parse the final decision
    decision_agent = Agent(instruction='You are an answer parser.', role='Answer Parser', model_info='gpt-4o-mini')
    decision_agent.chat('You are an answer parser.')
    decision_answer, decision_answer_usage = decision_agent.chat(f'The following are multiple choice questions (with answers) about medical knowledge.\n\nHere is the question: {question}\n\nOnly output the single letter from the following {final_decision}.', img_path=None)
    total_usage['prompt_tokens'] += decision_answer_usage['prompt_tokens']
    total_usage['completion_tokens'] += decision_answer_usage['completion_tokens']
    
    return {
        'majority': final_decision,
        'answer': decision_answer
    }, total_usage

def process_advanced_query(question, model, args):
    print("[STEP 1] Recruitment")
    group_instances = []
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model)
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited, recruited_usage = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following is an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format without any markdown symbols.")
    total_usage['prompt_tokens'] += recruited_usage['prompt_tokens']
    total_usage['completion_tokens'] += recruited_usage['completion_tokens']
    
    print(recruited)
    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]
    print(group_strings)
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()
        
        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, model=model)
        group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment, init_assessment_usage = group_instance.interact(comm_type='internal')
            total_usage['prompt_tokens'] += init_assessment_usage['prompt_tokens']
            total_usage['completion_tokens'] += init_assessment_usage['completion_tokens']
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment, assessment_usage = group_instance.interact(comm_type='internal')
            total_usage['prompt_tokens'] += assessment_usage['prompt_tokens']
            total_usage['completion_tokens'] += assessment_usage['completion_tokens']
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision, decision_usage = group_instance.interact(comm_type='internal')
            total_usage['prompt_tokens'] += decision_usage['prompt_tokens']
            total_usage['completion_tokens'] += decision_usage['completion_tokens']
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    tmp_agent = Agent(instruction=decision_prompt, role='Decision Maker', model_info=model)
    tmp_agent.chat(decision_prompt)

    final_decision, final_decision_usage = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", img_path=None)
    total_usage['prompt_tokens'] += final_decision_usage['prompt_tokens']
    total_usage['completion_tokens'] += final_decision_usage['completion_tokens']

    # Parse the final decision
    decision_agent = Agent(instruction='You are an answer parser.', role='Answer Parser', model_info='gpt-4o-mini')
    decision_agent.chat('You are an answer parser.')
    decision_answer, decision_answer_usage = decision_agent.chat(f'The following are multiple choice questions (with answers) about medical knowledge.\n\nHere is the question: {question}\n\nOnly output the single letter from the following {final_decision}.', img_path=None)
    total_usage['prompt_tokens'] += decision_answer_usage['prompt_tokens']
    total_usage['completion_tokens'] += decision_answer_usage['completion_tokens']
    
    return {
        'majority': final_decision,
        'answer': decision_answer
    }, total_usage
