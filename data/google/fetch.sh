#!/bin/zsh
# Fetch Google's two headline QEC datasets from Zenodo, resumable, verified.
set -u
cd "$(dirname "$0")"
small_url="https://zenodo.org/api/records/6804040/files/google_qec3v5_experiment_data.zip/content"
big_url="https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content"
big_size=5716907033
parts=4
chunk=$(( (big_size + parts - 1) / parts ))
curl -sfL --retry 10 -C - -o google_qec3v5_experiment_data.zip "$small_url" &
for i in $(seq 0 $((parts - 1))); do
  start=$(( i * chunk )); end=$(( start + chunk - 1 )); (( end >= big_size )) && end=$(( big_size - 1 ))
  want=$(( end - start + 1 ))
  (
    while true; do
      have=$( [ -f part$i ] && stat -f %z part$i || echo 0 )
      (( have >= want )) && break
      curl -sfL --retry 10 -r $(( start + have ))-$end "$big_url" >> part$i || sleep 5
    done
  ) &
done
wait
cat part0 part1 part2 part3 > google_105Q_surface_code_d3_d5_d7.zip && rm part0 part1 part2 part3
echo "md5 small $(md5 -q google_qec3v5_experiment_data.zip) expected a7fd8b481c3087090093106382dc217d"
echo "md5 big   $(md5 -q google_105Q_surface_code_d3_d5_d7.zip) expected 21fa6ad35b395d838ebcdbc92e364a12"
