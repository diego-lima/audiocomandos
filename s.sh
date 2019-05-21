#!/bin/sh
renomear(){
num=1
for file in *.wav; do
       mv "$file" "$(printf "%u" $num).wav"
       let num=$num+1
done
}
# para arquivos que começam com dash:
# for file in -*; do mv -- "$file" "a#$file"; done
