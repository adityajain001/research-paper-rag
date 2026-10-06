from ingest import load_corpus, create_embeddings
from rag import retrieve
from sentence_transformers import SentenceTransformer
    

all_chunks = load_corpus('data')

model = SentenceTransformer("all-MiniLM-L6-v2") 
embeddings = create_embeddings(all_chunks, model)

evaluation_questions = [
    # FACTR
    {
        "question": "How does force feedback help improve contact-rich manipulation policies?",
        "expected_paper": "factr"
    },
    {
        "question": "Why is force information useful when learning manipulation tasks involving contact?",
        "expected_paper": "factr"
    },
    {
        "question": "How is curriculum learning used to train policies for contact-rich manipulation?",
        "expected_paper": "factr"
    },
    {
        "question": "How does the method encourage the policy to attend to force information during training?",
        "expected_paper": "factr"
    },
    {
        "question": "What challenges arise when learning manipulation skills that require precise physical contact?",
        "expected_paper": "factr"
    },

    # ViPRA
    {
        "question": "How can robot policies learn useful behavior from videos without action labels?",
        "expected_paper": "VIPRA"
    },
    {
        "question": "How does the method use human videos to improve robot manipulation policies?",
        "expected_paper": "VIPRA"
    },
    {
        "question": "What information can be extracted from large-scale video data for robot learning?",
        "expected_paper": "VIPRA"
    },
    {
        "question": "How can knowledge learned from human activity videos be transferred to robotic manipulation?",
        "expected_paper": "VIPRA"
    },
    {
        "question": "How does the approach deal with differences between human behavior in videos and robot actions?",
        "expected_paper": "VIPRA"
    },

    # ManipGen
    {
        "question": "How are local manipulation policies used to perform long-horizon tasks?",
        "expected_paper": "ManipGen"
    },
    {
        "question": "Why does the system use specialized local policies instead of learning one policy for an entire task?",
        "expected_paper": "ManipGen"
    },
    {
        "question": "How are training examples generated for learning different manipulation skills?",
        "expected_paper": "ManipGen"
    },
    {
        "question": "How does the system combine individual manipulation skills to complete multi-stage tasks?",
        "expected_paper": "ManipGen"
    },
    {
        "question": "How does the approach enable scalable learning of multiple manipulation behaviors?",
        "expected_paper": "ManipGen"
    },

    # Neural MP
    {
        "question": "How does the learned planner generate collision-free robot motions around obstacles?",
        "expected_paper": "Neural_MP"
    },
    {
        "question": "How is test-time optimization used to improve trajectories produced by the neural motion planner?",
        "expected_paper": "Neural_MP"
    },
    {
        "question": "How are obstacles represented and checked for collisions during motion planning?",
        "expected_paper": "Neural_MP"
    },
    {
        "question": "How does closed-loop execution help the motion planner respond to changes in the environment?",
        "expected_paper": "Neural_MP"
    },
    {
        "question": "What advantages does a learned motion-planning policy provide compared with conventional planning approaches?",
        "expected_paper": "Neural_MP"
    }
]


for query in evaluation_questions:
    top_indices = retrieve(query['question'], model, embeddings, 5)
    query['retrieved_papers'] = []
    for index in top_indices:
        query['retrieved_papers'].append(all_chunks[index]['paper'])

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0
for query in evaluation_questions:
    bool_list = []
    expected_paper = query['expected_paper']
    for retrieved_paper in query['retrieved_papers']:
        bool_list.append(expected_paper == retrieved_paper)
        
    if any(bool_list[:1]):
        recall_at_1+=1
    else:
        print("\nRecall@1 failures:")
        print ('question', query["question"])
        print ('expected_paper',expected_paper)
        print('retrieved_papers', query['retrieved_papers'])
        print('\n')
    
    if any(bool_list[:3]):
        recall_at_3+=1
    else:
        print("\nRecall@3 failures:")
        print ('question', query["question"])
        print ('expected_paper',expected_paper)
        print('retrieved_papers', query['retrieved_papers'])
        print('\n')
    
    if any(bool_list[:5]):
        recall_at_5+=1
    else:
        print("\nRecall@5 failures:")
        print ('question', query["question"])
        print ('expected_paper',expected_paper)
        print('retrieved_papers', query['retrieved_papers'])
        print('\n')

question_size = len(evaluation_questions)
recall_at_1 = recall_at_1/question_size
recall_at_3 = recall_at_3/question_size
recall_at_5 = recall_at_5/question_size

print(f"Recall@1: {recall_at_1:.2f}")
print(f"Recall@3: {recall_at_3:.2f}")
print(f"Recall@5: {recall_at_5:.2f}")