# MP3 Reflection

## Grounding & abstention
Where did your assistant abstain correctly? Did it ever answer something it
shouldn't have (or refuse something it could have answered)? What threshold did
you settle on, and why?

Assistant abstained corretly for following should_abstain=True for top-3,top-5 & top-7

"id": "abstain_moriarty"
"id": "abstain_out_of_domain"
"id": "q7_YOUR_ABSTAIN"

Following from answerable were abstained for top-3 but grounded for top-5 and top-7 which means that
relevant chunks were missed for top-3. 

q1_red_headed_league_assistant
q2_speckled_band_weapon    
q4_engineers_thumb_press   

Started with Threshold=0.3 top-k=3 but it was generating context noise resulting in relevant chunks missing for q4_engineers_thumb_press
Setting it to Threshold=0.5 for top-k=3 fetched the relevant chunk for q4_engineers_thumb_press improving the grounded score. 
Had to settled with Threshold=0.45 as beyond it coisine similarity search started filtering out relevant relevant chunks. 
With Threshold=0.45 with top-k=5 grounded got grounded score=1.

## Citations
Did the model ever cite a chunk that didn't support the claim, or invent a marker?
What did `validate_citations` catch?

Model did not cite invalid chunk that didn't support the claim or invent marker. 


## Reading your eval table
What do your four numbers say? Which is weakest, and what would you change?
Grounded=0.6 was weakest with top-k=3 and Threshold=0.45 as the relvant chunks were missed. 
Changing top-k=5 with Threshold=0.45 fetched relevant chunks which resulted in Grounded=1

id                              type              hit   abstained  grounded  ok
  --------------------------------------------------------------------------
  q1_red_headed_league_assistant  answerable         ✓          ✗          ✓         ✓
  q2_speckled_band_weapon         answerable         ✓          ✗          ✓         ✓
  q3_scandal_bohemia_adler        answerable         ✓          ✗          ✓         ✓
  q4_engineers_thumb_press        answerable         ✓          ✗          ✓         ✓
  abstain_moriarty                should_abstain     ✗          ✓          ✗         ✓
  abstain_out_of_domain           should_abstain     ✗          ✓          ✗         ✓
  q5_YOUR_TOPIC                   answerable         ✓          ✗          ✓         ✓
  q6_YOUR_TOPIC                   answerable         ✓          ✗          ✓         ✓
  q7_YOUR_ABSTAIN                 should_abstain     ✗          ✓          ✗         ✓
  --------------------------------------------------------------------------

  Retrieval hit rate    : 1.0   (6 answerable questions)
  Grounded & cited rate : 1.0   (KPI: grounded response rate)
  Abstention accuracy   : 1.0   (3 should-abstain questions)
  Hallucination rate    : 0.0   (KPI: hallucination rate)

## One surprise
Something from the build that genuinely surprised you.
```
For relevant chunks expectation was that Cosine similarity would on higher side around 0.8 but for most of relevant chunks also the Cosine similarity was coming below 06. 
When researched got to know that models like text-embedding-3-small or bge-large frequently score strong matches around 0.50 to 0.60
