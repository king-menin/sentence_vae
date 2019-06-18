#!/usr/bin/env bash
mkdir ru_data
cd ru_data
wget http://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2
cd ../
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python3 WikiExtractor.py -o ../data/wiki/ --no-templates --processes 8 ../ru_data/ruwiki-latest-pages-articles.xml.bz2
