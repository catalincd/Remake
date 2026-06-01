# Configs

Each YAML fully describes one run. Switch label space / subsample on the fly with
CLI overrides instead of new files, e.g.:

    ./run.sh train configs/lgbm_coarse.yaml --set label_space=flat75 name=lgbm_flat75
    ./run.sh train configs/cnn_bigru_coarse.yaml --set data.max_per_class=null   # full data

label_space:  coarse11 | flat75 | specialist:<group>   (groups: raw bitmap vector
video archive executable office published text audio other)
features:     '_'-joined groups, e.g. stats_hist (needs `./run.sh features`)
Subsample caps keep runs tractable on a 6750 XT; set to null for full data.
