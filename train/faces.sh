#!/bin/sh

if [ $# -ne 1 ]
then
  echo "Usage: $0 <images_dir>"
  exit 1
fi

rm faces.txt labels.txt 2>/dev/null

find "$1" -name "*.jpg" | sort | while read f
do
  l=$(detector "$f" 0.87)
  if [ -n "$l" ]
  then
    echo "$l\t$f"
    echo "$f" >>faces.txt
    echo "$l" >>labels.txt
  fi
done
