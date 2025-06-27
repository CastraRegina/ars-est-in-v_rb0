#!/bin/bash
# download fonts from fonts.google.com and unzip the *.ttf files in this folder

mkdir zips

# download by using wget does not work anymore:
#wget https://fonts.google.com/download?family=Dancing%20Script   -O zips/Dancing_Script.zip
#wget https://fonts.google.com/download?family=Grandstander       -O zips/Grandstander.zip
#wget https://fonts.google.com/download?family=Noto%20Sans%20Mono -O zips/Noto_Sans_Mono.zip
#wget https://fonts.google.com/download?family=Recursive          -O zips/Recursive.zip
#wget https://fonts.google.com/download?family=Roboto%20Flex      -O zips/Roboto_Flex.zip
#wget https://fonts.google.com/download?family=Roboto%20Mono      -O zips/Roboto_Mono.zip
#wget https://fonts.google.com/download?family=Cantarell          -O zips/Cantarell.zip
#wget https://fonts.google.com/download?family=Petrona            -O zips/Petrona.zip
#wget https://fonts.google.com/download?family=Caveat             -O zips/Caveat.zip


for i in zips/*.zip ; do
    unzip -o "${i}" "*.ttf" -x "static/*" 
done

