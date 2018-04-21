CURRENT_DATA := data/rap


.PHONY: oyomi
oyomi:
	cd $(CURRENT_DATA) && cat raw.txt | mecab -Oyomi > raw_yomi.txt


.PHONY: vector
vector:
	python parser.py
	python preprocess.py


.PHONY: train
train:
	python train.py


.PHONY: aws-up
aws-up:
	aws s3 sync ./data s3://nlp-concierge/data


.PHONY: aws-ls
aws-ls:
	aws s3 ls s3://nlp-concierge/data/ --recursive --human-readable --summalize