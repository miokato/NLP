CURRENT_DATA := data/rap


.PHONY: oyomi
oyomi:
	cd $(CURRENT_DATA) && cat raw.txt | mecab -Oyomi > raw_yomi.txt


.PHONY: vector
vector:
	python3 parser.py
	python3 preprocess.py


.PHONY: train
train:
	python3 train.py


.PHONY: aws-up
aws-up:
	aws s3 sync ./data s3://nlp-concierge/data


.PHONY: aws-ls
aws-ls:
	aws s3 ls s3://nlp-concierge/data/ --recursive --human-readable --summalize


.PHONY: aws-down
aws-down:
	aws s3 sync s3://nlp-concierge/data ./data