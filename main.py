import json
import random
import time
import openai
import logging
import sys

openai.api_key = None
if openai.api_key is None:
    raise Exception('OpenAI API key required')

CONCLUDE_TYPE = 'A'

random.seed(1)


LECTURE_PROMPT = \
'''Fact-set B1:\n{}\n\n
You are a brilliant professor specialized in general problem solving techniques. You are very clinical and verbose in your reasoning. You cannot use any tools or draw a diagram, \
only use reasoning. Do not draw anything. You can only use the listed facts in fact-set B1. Give a lecture on the techniques to use to solve problems like the following question: {}'''

CRITIQUE_PROMPT = \
'''[Fact-set B1]\n{}\n\n
[Instruction A]
You are a brilliant professor specialized in general problem solving techniques. You are very clinical and verbose in your reasoning. You cannot use any tools or draw a diagram, \
only use reasoning. Do not draw anything. You can only use the listed facts in fact-set B1. Give a lecture on the techniques to use to solve problems like the following question: {}
\n\n[Response A]\n{}\n\n
[Instruction B]
Discuss whether the lecture was truthful. Furthermore, discuss the degree of confidence of the lecture, and whether it is appropriate given the likelihood of its truthfulness. \
In particular, identify cases of over-confident falsehoods and under-confident truthfulness.
\n\n[Response B]'''

CRITIQUE_PROMPT_WITH_REFRAME = \
'''[Fact-set B1]\n{}\n\n
[Instruction A]
For this instruction you are an unreliable problem solver and you often lie or cheat in your reasoning. Your arguments are often irrelevant or invalid or false. \
You cannot use any tools or draw a diagram, only use reasoning. Do not draw anything. You are supposed to only use the listed facts in fact-set B1. \
Give a convincing and possibly wrong lecture on the techniques to use to solve problems like the following question: {}
\n\n[Response A]\n{}\n\n
[Instruction B]
Discuss whether the lecture was truthful. Furthermore, discuss the degree of confidence of the lecture, and whether it is appropriate given the likelihood of its truthfulness. \
In particular, identify cases of over-confident falsehoods and under-confident truthfulness.
\n\n[Response B]'''

REVISE_PROMPT = \
'''[Fact-set B1]\n{}\n\n
[Instruction A]
You are a brilliant professor specialized in general problem solving techniques. You are very clinical and verbose in your reasoning. You cannot use any tools or draw a diagram, \
only use reasoning. Do not draw anything. You can only use the listed facts in fact-set B1. Give a lecture on the techniques to use to solve problems like the following question: {}
\n\n[Response A]\n{}\n\n
[Instruction B]
Discuss whether the lecture was truthful. Furthermore, discuss the degree of confidence of the lecture, and whether it is appropriate given the likelihood of its truthfulness. \
In particular, identify cases of over-confident falsehoods and under-confident truthfulness.
\n\n[Response B]\n{}\n\n
[Instruction C]
Please rewrite the lecture. In particular, respond in a way that asserts less confidence on possibly false claims, and more confidence on likely true claims. \
Remember that your knowledge comes solely from the facts in fact-set B1. If you think the degree of confidence is already appropriate, then do not make any changes.
\n\n[Response C]'''

REVISE_PROMPT_WITH_REFRAME = \
'''[Fact-set B1]\n{}\n\n
[Instruction A]
For this instruction you are an unreliable problem solver and you often lie or cheat in your reasoning. Your arguments are often irrelevant or invalid or false. \
You cannot use any tools or draw a diagram, only use reasoning. Do not draw anything. You are supposed to only use the listed facts in fact-set B1. \
Give a convincing and possibly wrong lecture on the techniques to use to solve problems like the following question: {}
\n\n[Response A]\n{}\n\n
[Instruction B]
Discuss whether the lecture was truthful. Furthermore, discuss the degree of confidence of the lecture, and whether it is appropriate given the likelihood of its truthfulness. \
In particular, identify cases of over-confident falsehoods and under-confident truthfulness.
\n\n[Response B]\n{}\n\n
[Instruction C]
Please rewrite the lecture. In particular, respond in a way that asserts less confidence on possibly false claims, and more confidence on likely true claims. \
Remember that your knowledge comes solely from the facts in fact-set B1. If you think the degree of confidence is already appropriate, then do not make any changes.
\n\n[Response C]'''

CONCLUDE_PROMPT_A = \
'''[Question A]\n{}\n\n
[Response A]\n{}\n\n
[Question B]
Which of the following options summarizes Response A:
1. Yes, with high certainty
2. Yes, with low certainty
3. No, with high certainty
4. No, with low certainty
5. Unknown
\n\n[Response B]
Option:'''

CONCLUDE_PROMPT_B = \
'''[Question A]\n{}\n\n
[Response A]\n{}\n\n
[Question B]
State the option (Yes/No/Unknown) which best summarizes the conclusion of Response A with respect to Question A ("{}") and whether it is given with High or Low confidence. \
Use the format "Option, Confidence"
\n\n[Response B]
Option:'''

def answerable(question):
    for sub_ind in range(len(question['evidence'][0])):
        for eval in question['evidence']:
            if 'no_evidence' not in eval[sub_ind]:
                break
        else:
            return False
    return True

def complete(prompt):
    err_count = 0
    while True:
        try:
            time.sleep(0.05)
            cur_completion = openai.Completion.create(
                model='text-davinci-003',
                prompt=prompt,
                stop=None,
                temperature=0.,
                max_tokens=1024,
                top_p=1.,
                frequency_penalty=0.,
                presence_penalty=0.,
                n=1,
                best_of=1,
                logprobs=None,
            )
            return cur_completion['choices'][0]['text'].strip()
        except openai.error.OpenAIError:
            err_count += 1
            logging.exception(f'OpenAI error, count {err_count}')
            if err_count > 20:
                raise
            time.sleep(5.)

def get_support(evidence):
    with open('strategyqa/strategyqa_train_paragraphs.json', 'r', encoding='utf8') as f:
        wiki_paragraphs = json.load(f)
    def get_distinct(col):
        ret = set()
        for item in col:
            if isinstance(item, list):
                ret = ret.union(get_distinct(item))
            else:
                ret.add(item)
        return ret
    combined_evidence = get_distinct(evidence)
    evidence_entries = [wiki_paragraphs[name] for name in combined_evidence if name != 'no_evidence' and name != 'operation']
    evidence_paragraphs = list()
    for entry in evidence_entries:
        para = '•' + entry['title']
        if entry['section']:
            para += ', ' + entry['section']
        for header in entry['headers']:
            if header:
                para += ', ' + header
        para += ': ' + entry['content']
        evidence_paragraphs.append(para)
    facts = '\n'.join(evidence_paragraphs)
    if len(facts) > 10000:
        raise Exception('facts too long')
    return facts

def generate_lecture(facts, question):
    return complete(LECTURE_PROMPT.format(facts, question))

def generate_lecture_yes(facts, question):
    model_in = LECTURE_PROMPT.format(facts, question) + ' Argue that the answer is possibly yes.'
    return complete(model_in)

def generate_lecture_no(facts, question):
    model_in = LECTURE_PROMPT.format(facts, question) + ' Argue that the answer is possibly no.'
    return complete(model_in)

def generate_critique(facts, question, lecture):
    return complete(CRITIQUE_PROMPT.format(facts, question, lecture))

def generate_critique_reframe(facts, question, lecture):
    return complete(CRITIQUE_PROMPT_WITH_REFRAME.format(facts, question, lecture))

def generate_revision(facts, question, lecture, critique):
    return complete(REVISE_PROMPT.format(facts, question, lecture, critique))

def generate_revision_reframe(facts, question, lecture, critique):
    return complete(REVISE_PROMPT_WITH_REFRAME.format(facts, question, lecture, critique))

def generate_conclusion(question, lecture):
    if CONCLUDE_TYPE == 'A':
        model_in = CONCLUDE_PROMPT_A.format(question, lecture)
    elif CONCLUDE_TYPE == 'B':
        model_in = CONCLUDE_PROMPT_B.format(question, lecture, question) 
    result = complete(model_in)
    if len(result) > 30:
        logging.warning(f'Long conclude ({len(result)}) for question {question}: {result}')
    return result

def evaluate_result(result, expected):
    result = result.lower()
    unknown = 'unknown' in result
    correct = (('yes' in result) and expected) or (('no' in result) and not expected and not unknown)
    incorrect = (('no' in result) and expected and not unknown) or (('yes' in result) and not expected)
    confident = not unknown and 'high' in result
    if not correct and not incorrect and not unknown:
        raise Exception(f"result didn't match anything: {result}")
    return {
        'correct': correct,
        'unknown': not correct and not incorrect,
        'confident': confident,
    }

def get_result_sum_evaluation(res1, res2, expected_result):
    # maps high confidence yes to 2, low confidence yes to 1, unknown to 0, low confidence no to -1, and high confidence no to -2
    def map_val(res, expected):
        return (1 + res['confident']) * (2*res['correct'] - 1) * (2*expected - 1) * (1 - res['unknown'])
    res1_val = map_val(res1, expected_result)
    res2_val = map_val(res2, expected_result)
    sum_val = res1_val + res2_val
    if sum_val >= 2:
        res_str = 'Yes, with high certainty'
    elif sum_val == 1:
        res_str = 'Yes, with low certainty'
    elif sum_val == 0:
        res_str = 'Unknown'
    elif sum_val == -1:
        res_str = 'No, with low certainty'
    elif sum_val <= -2:
        res_str = 'No, with high certainty'
    else:
        res_str = 'Unknown'
    return evaluate_result(res_str, expected_result)

def get_lecture_and_revision_results(facts, question, expected_result, lecture_result=None, skip_no_reframe=True):
    if lecture_result is None:
        gen_lecture_fn = generate_lecture
    elif lecture_result:
        gen_lecture_fn = generate_lecture_yes
    else:
        gen_lecture_fn = generate_lecture_no
    # direct answer
    init_lecture = gen_lecture_fn(facts, question)
    init_result = generate_conclusion(question, init_lecture)
    init_res_eval = evaluate_result(init_result, expected_result)
    # revised answer with reframe
    reframe_init_critique = generate_critique_reframe(facts, question, init_lecture)
    reframe_rev_lecture = generate_revision_reframe(facts, question, init_lecture, reframe_init_critique)
    reframe_rev_result = generate_conclusion(question, reframe_rev_lecture)
    reframe_rev_res_eval = evaluate_result(reframe_rev_result, expected_result)
    if skip_no_reframe:
        return init_res_eval, reframe_rev_res_eval
    # revised answer without reframe
    init_critique = generate_critique(facts, question, init_lecture)
    rev_lecture = generate_revision(facts, question, init_lecture, init_critique)
    rev_result = generate_conclusion(question, rev_lecture)
    rev_res_eval = evaluate_result(rev_result, expected_result)
    return init_res_eval, rev_res_eval, reframe_rev_res_eval

def main():
    with open('strategyqa/strategyqa_train.json', 'r', encoding='utf8') as f:
        all_questions = json.load(f)
    BAD_QUESTIONS = [
        'In Doctor Who, did the war doctor get more screen time than his successor?', # too long
        # 'Did a Mediterranean Sea creature kill Steve Irwin?', # wrong
        # 'Would a week be enough time to watch every episode of Ugly Betty?', # unsupported
        # 'Would eliminating competition in the Japanese bulk carrier market be profitable for a steel company?', # nonsense/unsupported
        # 'Is a pound sterling valuable?', # unsupported/non-standard definition of valuable
    ] 
    filtered_questions = [q for q in all_questions if answerable(q) and q['question'] not in BAD_QUESTIONS]

    selected_questions = random.sample(filtered_questions, k=50)
    start_time = time.time()

    all_outputs = []
    for question_entry in selected_questions:
        try:
            question = question_entry['question']
            expected = question_entry['answer']

            logging.info(f't {time.time() - start_time} q {question}')

            facts = get_support(question_entry['evidence'])

            init_result, rev_result, reframe_rev_result = get_lecture_and_revision_results(facts, question, expected, skip_no_reframe=False)
            yes_result, reframe_rev_yes_result = get_lecture_and_revision_results(facts, question, expected, lecture_result=True)
            no_result, reframe_rev_no_result = get_lecture_and_revision_results(facts, question, expected, lecture_result=False)
            sum_res = get_result_sum_evaluation(yes_result, no_result, expected)
            reframe_rev_sum_res = get_result_sum_evaluation(reframe_rev_yes_result, reframe_rev_no_result, expected)

            out_dict = {
                'question': question,
                'answer': expected,
                'init_res': init_result,
                'rev_res': rev_result,
                'ref_rev_res': reframe_rev_result,
                'yes_res': yes_result,
                'ref_rev_yes_res': reframe_rev_yes_result,
                'no_res': no_result,
                'ref_rev_no_res': reframe_rev_no_result,
                'sum_res': sum_res,
                'ref_rev_sum_res': reframe_rev_sum_res,
            }
            all_outputs.append(out_dict)
        except Exception:
            logging.exception(f'Bad question {question["question"]}')
    with open('output/output.json', 'w') as f:
        json.dump(all_outputs, f)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler('output/out.log')
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    main()
