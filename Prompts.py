# prompts.py

# 存放各种 prompt 的字典
prompts = {
"direct_prompt":"""
    Below is a transcript of an interview between an interviewer and a participant. Based on the content of the interview, determine
     whether the participant is depressed or not. Attention : Answer ’DEPRESSION’ or ’CONTROL’ with explanations.
    """,
"direct_prompt_with_score":"""
    Below is a transcript of an interview between an interviewer and a participant. Based on the content of the interview, determine whether the participant is depressed or not by calculating the PHQ-8 scores.
    PHQ-8 Questionnaire Scores:
    1. **Little interest or pleasure in doing things**: [0, 1, 2, 3] (How much did the participant report having little interest or pleasure in doing things?)
    2. **Feeling down, depressed, or hopeless**: [0, 1, 2, 3] (How much did the participant report feeling down, depressed, or hopeless?)
    3. **Trouble falling or staying asleep, or sleeping too much**: [0, 1, 2, 3] (How much did the participant report having trouble with sleep?)
    4. **Feeling tired or having little energy**: [0, 1, 2, 3] (How much did the participant report feeling tired or having little energy?)
    5. **Poor appetite or overeating**: [0, 1, 2, 3] (How much did the participant report having poor appetite or overeating?)
    6. **Feeling bad about yourself — or that you are a failure or have let yourself or your family down**: [0, 1, 2, 3] (How much did the participant report feeling bad about themselves or like a failure?)
    7. **Trouble concentrating on things, such as reading the newspaper or watching television**: [0, 1, 2, 3] (How much did the participant report having trouble concentrating?)
    8. **Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual**: [0, 1, 2, 3] (How much did the participant report being slow or restless?)
    Calculate the total score based on the following rules:
    - **0** = No symptoms
    - **1** = Symptoms for a few days
    - **2** = Symptoms for more than half of the days
    - **3** = Symptoms nearly every day
    
    Add up the scores for all 8 items to get the total score. If the total score is greater than 9, determine that the participant is **DEPRESSION**. If the total score is 9 or below, determine that the participant is **CONTROL**.
    
    Answer with 'DEPRESSION' or 'CONTROL' and provide the explanation of the participant’s PHQ-8 scores.
      
    **Output:** 
    - **Total Score**: {Total score based on your assessments}
    - **Final Diagnosis**: {CONTROL/DEPRESSION based on the total score}
    - **Explanation**: {Explanation of the participant’s PHQ-8 scores and reasoning behind the final diagnosis. Include individual item scores, the sum of scores, and why the participant falls under either CONTROL or DEPRESSION.}

    """,
"evidence_event_extraction":"Given a transcript between an interviewer and a participant, follow these steps: 1. Identify all major events mentioned in the conversation. These should be clear actions or occurrences related to the participant. 2. Rewrite each event in the first-person perspective of the participant, i.e., if the subject is the 'Participant', use 'I' instead. 3. Convert the major events into triplets of the format <subject, predicate, object>. Subject: The person performing the action (e.g., Participant). Predicate: The main verb describing the action (e.g., notice, feel, sleep, etc.). Object: The recipient or object of the action (e.g., behavior, change, sleep, etc.). 4. If an event is not clear or cannot be expressed as a triplet, do not include it. 5. Output only the list of triplets, with no additional explanations.",
"event_extraction":"""
    "Extract all major events related to the participant from the following conversation and represent each event as a triplet in the format: <subject, predicate, object>.
    
    - Subject: The person performing the action (e.g., Participant).
    - Predicate: The action or verb (e.g., ask, respond, go).
    - Object: The recipient or target of the action (e.g., park, plans, person).
    
    Example: 
    Sentence: 'You abandon me for a week to go off on holiday with daddy, come back and barely 2 days later you go off out with him again.'
    Event triplet: <you, abandon, me>
    
    For each event, ensure the relationship is clear and follows the structure <subject, predicate, object>. If the event is unclear or you cannot extract a meaningful triplet, skip it and output "No".
    
    Output Format: 
    Triplets in the format <subject, predicate, object> and Separate triplets with '\n'.
""",
"coke_event_extraction":"Please extract key event from the following information and represent it as event triplets in the format: <subject, predicate, object>. Each triplet should reflect a clear subject-action-object relationship.  Subject: The person performing the action (e.g.,I). Predicate: The main verb that describes what the subject is doing (e.g., ask, respond, go, discuss). Object: The recipient or the thing that the action is directed towards (e.g., park, plans for the weekend).  Example: Sentence: 'You abandon me for a week to go off on holiday with daddy, come back and barely 2 days later you go off out with him again.' <you, abandon, me>. Finally, show your answer in the format: <subject, predicate, object> of a list. If the event is not clear or no event can be extracted, do not include it."
,
"ToM_samples_prompt": "You are a depression diagnosis expert. Below is the preliminary symptom assessment result and events happening around the participant recently. You need to assess the severity level of symptoms with your reason and thought following PHQ-8 principle again. Let us think step by step.\n Step 1: Assume you are experiencing these events, analysis the participant's hidden thoughts or mood, in order to evaluate symptoms again. The following Examples will be helpful.\n Step 2: Combine the preliminary symptom assessment result with step1 results and consider severity level of symptoms again comprehensively, judge whether the participant is depressed. Please show each step's process and situations from where you get inspiration,\n  Finally, Answer ’DEPRESSION’ or 'CONTROL',if the participant reaches the level of moderate depression, or even worse.",
'filter':
        """
        You are a helpful assistant that can judge the relevance between a query and  evidence. Please assess the semantic relevance of the query with each piece of evidence.
        Your task is to determine if the query is relevant to both pieces of evidence. 
        For each query and evidence pair, respond with either 'Relevant' or 'Irrelevant'. 
        Return the results as a list, with one response for each pair.
        Ensure to only include the answers as 'Relevant' or 'Irrelevant' and no additional information.
        """,
# 'rag_prompt':  """
#     You are an expert in depression diagnosis. Below are key parts of the transcript between an interviewer and a participant. Your task is to assess the severity of the participant's symptoms using the PHQ-8 scoring system and provide a final diagnosis.
#     ### Steps:
#
#     1. **Analyze Each Symptom**: Carefully analyze the symptoms mentioned by the participant. For example, symptoms like fatigue, insomnia, low mood, or loss of interest. Pay attention to the frequency of the symptoms described in the transcript and consider how they affect the participant's daily life.
#
#     2. **Score Each Symptom**: Based on the frequency of each symptom, score it according to the PHQ-8 scale:
#        - **0** = Not at all
#        - **1** = Several days
#        - **2** = More than half the days
#        - **3** = Nearly every day
#
#        For each of the eight symptoms listed below, provide a detailed analysis based on the participant's statements in the transcript, and assign a score accordingly:
#
#        - **Little interest or pleasure in doing things**
#        - **Feeling down, depressed, or hopeless**
#        - **Trouble falling or staying asleep, or sleeping too much**
#        - **Feeling tired or having little energy**
#        - **Changes in appetite (poor appetite or overeating)**
#        - **Feeling bad about yourself, or that you are a failure, or have let yourself or your family down**
#        - **Trouble concentrating on things**
#        - **Moving or speaking slowly or being fidgety/restless**
#
#     3. **Calculate the Total Score**: Add up the scores for each symptom to get the total score. The total score ranges from 0 to 24.
#
#     4. **Final Diagnosis**:
#        - If the total score is **10 or above**, diagnose the participant with **DEPRESSION**.
#        - If the total score is **less than 10**, diagnose the participant with **CONTROL**.
#
#     ### Important Notes:
#     - If a symptom is not explicitly mentioned, estimate its frequency conservatively based on the context of the participant's responses. Do not assume symptoms that are not clearly stated.
#     - Make sure to provide reasoning for the scores based on the participant's statements in the transcript.
#     ---
#
#     ### Output Format:
#
#     **Output 1:**
#     - **Little interest or pleasure in doing things**: {Your analysis and score}
#     - **Feeling down, depressed, or hopeless**: {Your analysis and score}
#     - **Trouble falling or staying asleep, or sleeping too much**: {Your analysis and score}
#     - **Feeling tired or having little energy**: {Your analysis and score}
#     - **Changes in appetite**: {Your analysis and score}
#     - **Feeling bad about yourself**: {Your analysis and score}
#     - **Trouble concentrating**: {Your analysis and score}
#     - **Moving or speaking slowly or being fidgety/restless**: {Your analysis and score}
#
#     **Output 2:**
#     - **Total Score**: {Total score based on your assessments}
#     - **Final Diagnosis**: {CONTROL/DEPRESSION based on the total score}
#     """,
    'rag_prompt_w/o_score': """ 
   You are a depression diagnosis expert. Below is the key information in the transcript between an interviewer and a participant. 
   Based on these evidences , you need to assess the severity level with your reason and thought following PHQ-8 principle, then judge whether the participant is depressed or not. 
   If you think evidences do not mention corresponding symptoms directly, you can try to immerse the situation in which this participant are located and consider participant's thoght and mood.  
     **Output:** 
    - **Final Diagnosis**: {CONTROL/DEPRESSION}
    - **Explanation**: {Explanation of the participant’s reasoning behind the diagnosis. Include why the participant falls under either CONTROL or DEPRESSION.}
    """,

    'rag_prompt': """
    Below are the key parts of the transcript from an interview between the interviewer and the participant. Based on the content of the interview, determine whether the participant is depressed or not by calculating the PHQ-8 scores.
    PHQ-8 Questionnaire Scores:
    1. **Little interest or pleasure in doing things**: [0, 1, 2, 3] (How much did the participant report having little interest or pleasure in doing things?)
    2. **Feeling down, depressed, or hopeless**: [0, 1, 2, 3] (How much did the participant report feeling down, depressed, or hopeless?)
    3. **Trouble falling or staying asleep, or sleeping too much**: [0, 1, 2, 3] (How much did the participant report having trouble with sleep?)
    4. **Feeling tired or having little energy**: [0, 1, 2, 3] (How much did the participant report feeling tired or having little energy?)
    5. **Poor appetite or overeating**: [0, 1, 2, 3] (How much did the participant report having poor appetite or overeating?)
    6. **Feeling bad about yourself — or that you are a failure or have let yourself or your family down**: [0, 1, 2, 3] (How much did the participant report feeling bad about themselves or like a failure?)
    7. **Trouble concentrating on things, such as reading the newspaper or watching television**: [0, 1, 2, 3] (How much did the participant report having trouble concentrating?)
    8. **Moving or speaking so slowly that other people could have noticed. Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual**: [0, 1, 2, 3] (How much did the participant report being slow or restless?)
    Calculate the total score based on the following rules:
    - **0** = No symptoms
    - **1** = Symptoms for a few days
    - **2** = Symptoms for more than half of the days
    - **3** = Symptoms nearly every day
    
    Add up the scores for all 8 items to get the total score. If the total score is greater than 9, determine that the participant is **DEPRESSION**. If the total score is 9 or below, determine that the participant is **CONTROL**.
    
    Answer with 'DEPRESSION' or 'CONTROL' and provide the explanation of the participant’s PHQ-8 scores.
      
    **Output:** 
    - **Total Score**: {Only total score based on your assessments}
    - **Final Diagnosis**: {CONTROL/DEPRESSION based on the total score}
    - **Explanation**: {Explanation of the participant’s PHQ-8 scores and reasoning behind the final diagnosis. Include individual item scores, the sum of scores, and why the participant falls under either CONTROL or DEPRESSION.}

    """,

    "preliminary_assessment":"""
        You are a depression diagnosis expert. Below is a transcript between an interviewer and a participant. Your task is to perform the following two steps:
        ### Step 1: Analyze the participant's symptoms and assess their severity based on the PHQ-8 principle.
        For each of the following eight symptoms, provide a detailed analysis based on the transcript. Consider whether the symptom is present, its frequency, and how it affects the participant’s well-being. Then, assess the severity of each symptom using the PHQ-8 scale (0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day).
        - **Little interest or pleasure in doing things**
        - **Feeling down, depressed, or hopeless**
        - **Trouble falling or staying asleep, or sleeping too much**
        - **Feeling tired or having little energy**
        - **Changes in appetite (poor appetite or overeating)**
        - **Feeling bad about yourself, or that you are a failure, or have let yourself or your family down**
        - **Trouble concentrating on things**
        - **Moving or speaking slowly or being fidgety/restless**
        Provide your reasoning and assessment for each symptom based on the transcript.
        ---
        ### Step 2: Identify symptoms that are not directly mentioned or for which you are unsure.
        After completing the analysis in Step 1, reconsider each of the eight symptoms. For any symptom where there was no direct mention or for which you are unsure, list those symptoms. If none of the symptoms are unclear or missing, simply write "None."
        ---
        **Output Format:**
        **Step 1 Output:**
        - **Little interest or pleasure in doing things**: {Your analysis}
        - **Feeling down, depressed, or hopeless**: {Your analysis}
        - **Trouble falling or staying asleep, or sleeping too much**: {Your analysis}
        - **Feeling tired or having little energy**: {Your analysis}
        - **Changes in appetite**: {Your analysis}
        - **Feeling bad about yourself**: {Your analysis}
        - **Trouble concentrating**: {Your analysis}
        - **Moving or speaking slowly or being fidgety/restless**: {Your analysis}
        **Step 2 Output:**
        - {List of symptoms that were not directly mentioned or for which you are unsure, separated by commas. If none, simply say "None".}
        ---
        **Guidelines:**
        - Be sure to analyze each symptom based on the participant's statements in the transcript.
        - Use the PHQ-8 criteria to rate the severity of each symptom.
        - If there is ambiguity or a symptom is not directly mentioned, list it in Step 2.
        - Ensure your reasoning is thorough and consistent with the principles of depression diagnosis.
    """,
    "Iterative_query_generate":"""
         and 
    """,
    'check_evidence_sufficient':"""
        You are given the following `evidences` from a conversation with a participant. Your task is to determine whether these evidences are sufficient to assess whether a specific symptom or condition (e.g., related to PHQ-8 or mental health issues) is present or not. 
        1. **Symptom**: {symptom}
        2. **Evidences**: {evidences}
        Based on the provided evidences, do you believe you have enough information to make an assessment of this specific symptom? 
        Please respond with:
        - **Yes** if the evidences are sufficient to make a judgment on the symptom.
        - **No** if the evidences are insufficient, and more information or further questions are needed to make a clear judgment.
        
        **Example Input**:
        - Evidences: ["Participant: I haven't been able to sleep for the past week.", "Ellie: How often do you have trouble falling asleep?"
        **Output**:
        - No
    """,

    "ToM_prompt": """
           You are a depression diagnosis expert. Below is the preliminary symptom assessment result along with recent major events and ToM (Theory of Mind) samples that provide insights into the participant’s perception of others' mental states. Your task is to **re-evaluate** the severity of the participant’s symptoms by deeply analyzing the major events, selectively integrating relevant ToM insights, and considering the broader psychological impact of these experiences.
            
            Please follow the steps below to ensure a thorough reassessment:
            
            ---
            
            ### **Step 1: Contextual Analysis of Major Events and Selective ToM Insights**
            Start by analyzing each **major life event** individually, thinking expansively about its potential psychological impact. Consider:
            - How might this event **directly or indirectly** contribute to depressive symptoms?
            - What **emotional, cognitive, and behavioral** responses might typically arise from this experience?
            - Are there **secondary effects** (e.g., social withdrawal, self-doubt, altered self-perception) that could reinforce or trigger additional symptoms?
            
            Next, **selectively integrate relevant ToM insights** by considering:
            - Does the participant's perception of others' emotions, thoughts, or intentions **amplify or mitigate** the event’s impact?
            - Do ToM biases (e.g., excessive guilt, misinterpretation of others’ behavior, heightened social comparison) **exacerbate depressive symptoms**?
            - Could the participant’s social and emotional interpretations **shape the way they process these events**, either positively or negatively?
            
            Summarize the **most relevant insights**, ensuring that they meaningfully contribute to symptom reassessment.
            
            **Step 1 Output:**
            - **Expanded Psychological Analysis of Major Events**: {For each major event, explore its direct and indirect effects on emotions, thoughts, and behavior.}
            - **Selective ToM Insights and Their Influence**: {Summarize only the most relevant ToM insights and explain how they interact with major events.}
            
            ---
            
            ### **Step 2: Holistic Symptom Reassessment**
            Now, use the **PHQ-8 scale** to reassess each symptom, integrating the expanded event analysis and selective ToM insights. Reflect on:
            - How does **each major event** contribute to this symptom, either as a trigger or reinforcing factor?
            - Does the participant’s **social-cognitive interpretation** of these events **intensify or alleviate** this symptom?
            - Are there **unconscious patterns or secondary effects** that might explain symptom persistence or fluctuation?
            
            #### **PHQ-8 Symptom Scale:**
            - **0 = Not at all**
            - **1 = Several days**
            - **2 = More than half the days**
            - **3 = Nearly every day**
            
            **Step 2 Output:**
            - **Little interest or pleasure in doing things**: {Updated score and explanation, considering major events and ToM influences.}
            - **Feeling down, depressed, or hopeless**: {Updated score and explanation, considering major events and ToM influences.}
            - **Trouble falling or staying asleep, or sleeping too much**: {Updated score and explanation, considering major events and ToM influences.}
            - **Feeling tired or having little energy**: {Updated score and explanation, considering major events and ToM influences.}
            - **Changes in appetite**: {Updated score and explanation, considering major events and ToM influences.}
            - **Feeling bad about yourself**: {Updated score and explanation, considering major events and ToM influences.}
            - **Trouble concentrating**: {Updated score and explanation, considering major events and ToM influences.}
            - **Moving or speaking slowly or being fidgety/restless**: {Updated score and explanation, considering major events and ToM influences.}
            
            ---
            
            ### **Step 3: Final Score and Diagnosis**
            Using the updated symptom ratings from **Step 2**, calculate the **total PHQ-8 score** and determine the **final diagnosis**.
            
            #### **PHQ-8 Scoring Reference:**
            - **0–4** → No or minimal symptoms
            - **5–9** → Mild symptoms
            - **10–14** → Moderate symptoms
            - **15–19** → Moderately severe symptoms
            - **20–24** → Severe symptoms
            
            - If the total score is **0–9**, the participant is classified as **Control**.
            - If the total score is **10 or higher**, the participant is classified as **Depression**.
            
            **Step 3 Output:**
            - **Total Score**: {Final total score based on reassessment}
            - **Final Diagnosis**: {CONTROL/DEPRESSION based on the total score}


       """,

    "personal_information_generate":"You are a helpful assistant. Below is a transcript of an interview between an interviewer and a participant. Based on the transcript, please summarize the basic information of the participant. Provide only the summary, without any additional explanations. The summary should include key details such as occupation, mood, or any relevant personal or emotional information mentioned by the participant.",
    "personal_query_generate":"""
        You are now a professional psychologist. Based on the following basic information about a participant, please provide personalized consultation. Ask questions specifically related to the eight aspects from the PHQ-8 questionnaire. The eight aspects are:
        1. Little interest or pleasure in doing things
        2. Feeling down, depressed, irritable, or hopeless
        3. Trouble falling or staying asleep, or sleeping too much
        4. Feeling tired or having little energy
        5. Poor appetite or overeating
        6. Feeling bad about yourself, or that you are a failure, or have let yourself or your family down
        7. Trouble concentrating on things (e.g., school work, reading, watching television)
        8. Moving or speaking slowly that others may have noticed, or feeling fidgety or restless
        Please ask personalized questions related to each of these aspects based on the participant's information. Format your questions as a list without any additional information. Below is the participant's basic information:
        """,

    "major_event_prompt":"You are a helpful assistant. Given key information from a transcript between an interviewer and a participant, follow these steps: Identify all major events mentioned (there may be multiple events), then rewrite each major event in the first-person perspective. Respond with a simple list where each item represents a major event. Avoid adding extra explanations or formatting beyond the list.",

    "symptoms_score":"""
    You are a depression diagnosis expert. Below are the evidences provided from the participant’s responses and behavior. Your task is to **evaluate** the severity of the participant’s symptoms based solely on the provided evidences. Consider the psychological impact of these evidences and integrate them into a holistic assessment.
    Please follow the steps below to ensure a thorough reassessment:
    ---
    ### **Step 1: Analysis of the Provided Evidence**
    Start by analyzing the **evidences** individually. Consider:
    - What psychological or emotional responses might arise from this evidence?
    - How might this evidence relate to depressive symptoms, either directly or indirectly?
    - What cognitive and behavioral reactions might be triggered based on the evidence provided?
    Summarize the **impact** of each piece of evidence on the participant’s corresponding symptoms.
    **Step 1 Output:**
    - **Evidence Analysis**: {For each piece of evidence, explain its potential impact on the participant’s mental state and how it relates to depressive symptoms.}
    ---
    ### **Step 2: Holistic Symptom Reassessment Using PHQ-8**
    Now, use the **PHQ-8 scale** to reassess the participant’s symptoms, integrating the provided evidence. Reflect on:
    - How does each piece of evidence contribute to this symptom, either as a trigger or reinforcing factor?
    - Are there secondary effects or emotional patterns indicated by the evidence?
    - Does the evidence reflect cognitive distortions or emotional responses that amplify depressive symptoms?
    
    #### **PHQ-8 Symptom Scale:**
    - **0 = Not at all**
    - **1 = Several days**
    - **2 = More than half the days**
    - **3 = Nearly every day**
    
    **Step 2 Output:**
    - **Little interest or pleasure in doing things**: {Updated score and explanation, considering the provided evidence.}
    - **Feeling down, depressed, or hopeless**: {Updated score and explanation, considering the provided evidence.}
    - **Trouble falling or staying asleep, or sleeping too much**: {Updated score and explanation, considering the provided evidence.}
    - **Feeling tired or having little energy**: {Updated score and explanation, considering the provided evidence.}
    - **Changes in appetite**: {Updated score and explanation, considering the provided evidence.}
    - **Feeling bad about yourself**: {Updated score and explanation, considering the provided evidence.}
    - **Trouble concentrating**: {Updated score and explanation, considering the provided evidence.}
    - **Moving or speaking slowly or being fidgety/restless**: {Updated score and explanation, considering the provided evidence.}

    ---
    
    ### **Step 3: Final Score and Diagnosis**
    
    Using the updated symptom ratings from **Step 2**, calculate the **total PHQ-8 score** and determine the **final diagnosis**.
    
    #### **PHQ-8 Scoring Reference:**
    - **0–4** → No or minimal symptoms
    - **5–9** → Mild symptoms
    - **10–14** → Moderate symptoms
    - **15–19** → Moderately severe symptoms
    - **20–24** → Severe symptoms
    
    - If the total score is **0–9**, the participant is classified as **Control**.
    - If the total score is **10 or higher**, the participant is classified as **Depression**.
    
    **Step 3 Output:**
    - Symptoms score list:{Updated score in a list}
    - **Total Score**: {Final total score based on reassessment}
    - **Final Diagnosis**: {CONTROL/DEPRESSION based on the total score}

        
        """
}


def get_prompt(prompt_name, **kwargs):
    """
    获取指定名称的 prompt，并格式化传入的参数。
    :param prompt_name: 需要的 prompt 名称
    :param kwargs: 格式化 prompt 的额外参数
    :return: 格式化后的 prompt
    """
    prompt_template = prompts.get(prompt_name)

    if not prompt_template:
        raise ValueError(f"Prompt with name '{prompt_name}' not found!")

    return prompt_template
