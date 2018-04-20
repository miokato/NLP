CURRENT_DATA := data/rap


.PHONY: oyomi
oyomi:
	cd $(CURRENT_DATA) && cat raw.txt | mecab -Oyomi > raw_yomi.txt