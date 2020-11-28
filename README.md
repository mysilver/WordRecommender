## WordRecommender to diversify crowdsourced paraphrases

### How to Run
- Required Services

(1) Running languagetool as a service.
Download lanuguage tool [https://languagetool.org/download/] ans run the following command in bash
```
nohup java -cp LanguageTool-4.2-SNAPSHOT/languagetool-server.jar org.languagetool.server.HTTPServer --port 5004 &
```

(2) Download "GoogleNews-vectors-negative300.bin" [https://github.com/mmihaltz/word2vec-GoogleNews-vectors] and put it in the root directory of the project. Next, run the follwoing bash command
```
nohup python3.5 word2vec_serivce.py &
```

(3) You can run "canonical utterance generator" as a service:
```shell script
pip3 install -r requirements.txt
python3 word_recommender.py
```

### Further Information
Please refer to  and cite the following article:

```
@inproceedings{mine_iui_2020,
author    = {Mohammad{-}Ali Yaghoub{-}Zadeh{-}Fard and
               Boualem Benatallah and
               Fabio Casati and
               Moshe Chai Barukh and
               Shayan Zamanirad},
  editor    = {Fabio Patern{\`{o}} and
               Nuria Oliver and
               Cristina Conati and
               Lucio Davide Spano and
               Nava Tintarev},
  title     = {Dynamic word recommendation to obtain diverse crowdsourced paraphrases
               of user utterances},
  booktitle = {{IUI} '20: 25th International Conference on Intelligent User Interfaces,
               Cagliari, Italy, March 17-20, 2020},
  pages     = {55--66},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3377325.3377486},
  doi       = {10.1145/3377325.3377486},
  timestamp = {Fri, 06 Mar 2020 14:10:34 +0100},
  biburl    = {https://dblp.org/rec/conf/iui/Yaghoub-Zadeh-Fard20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
