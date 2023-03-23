def main():
    print("""
    The model for calibration is provided in 'calibration.stan', and the data
    for calibration is provided in 'data_calibration.json'.

    The model has already been run, and samples from three chains are provided
    in fit_calibration-{1,2,3}.csv. Each chain was run with a warmup of 1000
    iterations, following which 1000 samples were taken with no thinning.

    If you wish to compile the model and run it yourself you are welcome, but
    this is not necessary unless you are changing the model and/or the data.

    If you do re-estimate, it is important to override the parameters in
    lynchsyndrome/experiments/common/parameters/data/endometrial.py as these
    will be used unless alternative instruction is given.
    """)


if __name__ == "__main__":
    main()
