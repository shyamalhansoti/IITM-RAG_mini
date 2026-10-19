Create mp2_reflection.md with these four sections:
# MP2 Reflection

## What worked
(One specific thing that worked — a chunking choice, a prompt detail, etc.)

Chunking,embedding and vecctor storage in qdrant worked as expected. 
System prompt had Citation(title+section) but learner_question.jsonl was expecting the source so the validation failed.
Validation was successful after corresponding changes. 

## What didn't work
(One specific failure mode you hit. Did chunks get too small? Did retrieval
pull the wrong section? Did the LLM hallucinate?)

Validation was failing as the System prompt mentioned title + section but the learner_questions.jsonl was 
validating with source("expected_source": "01_red_headed_league.txt") instead of title(The Red-Headed League). 
Inclusion of the source still returned the title as source. Explicitly providing in desription "source is the txt file" for source field in ANSWER_TOOL returned the proper source and validation was successful

Similarity score came same for top-3. Suspected that embeddings were identical but they were not.
Also checked if the Scalar Quantization is enabled but was not. 
Entire dataset consists of Sherlock Holmes stories
In a highly specialized dataset where every single text chunk shares the same characters, vocabulary, Victorian-era English style, and settings, vector space becomes incredibly crowded and concentrated which could be the contributing factor for same score. 

Next step to force distinct scores would be to try with following 

Hybrid Search(Vector+BM25)
Reranker

## What I'd change
(One thing you'd do differently with another 5 hours.)
Would attempt hybrid+rerank. 

## One surprise
(Anything from the build that genuinely surprised you. Could be technical —
"text-embedding-3-small was way better than I expected on rare names" —
or pedagogical — "I didn't realise how much hybrid retrieval was helping
until I tried dense-only here.")

LLM returned title instead of source in response. Just changing source description to "source is the txt file" 
response function call resolved the issue. Shows the importance of giving proper instructions.