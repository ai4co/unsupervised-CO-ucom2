ds_list=(
    "rand500"
    "rand800"
    "starbucks"
    "subway"
    "mcd"
)

for ds in "${ds_list[@]}"; do
    for i_seed in {0..4}; do
        python facility_location_scip.py --cfg "cfg/facility_location_${ds}.yaml" --timestamp ${i_seed};
    done
done

ds_list=(
    "rand500"
    "rand1000"
    "rail"
    "twitch"
)

for ds in "${ds_list[@]}"; do
    for i_seed in {0..4}; do
        python max_cover_scip.py --cfg "cfg/max_cover_${ds}.yaml" --timestamp ${i_seed};
    done
done
