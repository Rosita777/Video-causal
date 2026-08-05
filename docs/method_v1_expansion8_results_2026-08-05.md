# Method v1 eight-object expansion screen

## Result

The screen covers four mechanism families with two source objects each:

| Mechanism | Source objects | Strict valid |
| --- | --- | ---: |
| Liquid impact | water droplet, ice cube | 1/2 |
| Toppling collision | red rubber ball, wooden cylinder | 1/2 plus 1 borderline |
| Brittle fracture | wooden mallet, metal hammer | 2/2 |
| Powder impact | red apple, wooden cube | 2/2 |

Overall strict validity is 6/8, with two failures caused by pre-existing or ambiguous temporal evidence rather than missing objects. The six accepted source-object variants are sufficient for the next aligned-pair construction stage; the two borderline cases remain held-out feasibility notes and are not used for training yet.

## Decision

The object-diverse evaluation design is viable. Training should use accepted aligned pairs from multiple source objects within each mechanism family, while the failed waterdrop and ambiguous collision examples remain controls for the clean-source screening protocol.
