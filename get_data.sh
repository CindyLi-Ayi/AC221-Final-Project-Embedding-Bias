declare -a urls=('https://nlp.stanford.edu/data/glove.6B.zip'
                'https://nlp.stanford.edu/data/glove.42B.300d.zip'
                'https://nlp.stanford.edu/data/glove.840B.300d.zip'
                'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
)

data_dir=data
if [ -d ${data_dir} ]; then
  echo Directory ${data_dir} already exists, delete it.
#   rm -rf ${data_dir}
fi

mkdir ${data_dir}

i=0
for url in ${urls[@]}
do
  ((i=i+1))
  filename=$(printf "%02d" $i)
  echo =============== $filename $url ===============

  # download
  wget -c -O ${data_dir}/${filename}.zip $url

  # unzip
  echo Decompressing and Extracting ${data_dir}/${filename}.zip ......
  unzip ${data_dir}/${filename}.zip -d ${data_dir}

#   remove
  rm ${data_dir}/${filename}.zip

done

wget -c -O ${data_dir}/bad-words.txt https://www.cs.cmu.edu/~biglou/resources/bad-words.txt