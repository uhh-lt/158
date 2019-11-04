script_dir=$(dirname $0)
rsync -r "$script_dir/../../158_disambiguator/graph_vector/" ./graph_vector/
docker build -t wsd_index_builder ./
