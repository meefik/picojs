#!/bin/sh

if [ $# -ne 1 ]
then
  echo "Usage: $0 <images_dir>"
  exit 1
fi

rm backgrounds.txt 2>/dev/null

find "$1" -name "*.jpg" | sort | while read f
do
  l=$(detector "$f" 0.13)
  if [ -z "$l" ]
  then
    echo "$f"
    echo "$f" >>backgrounds.txt
  fi
done
