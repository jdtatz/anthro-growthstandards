from .gamlss_ext import FractionalPolynomial, BCPEModel

hcff = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-274.7848237296913,
        coefficients=(-149.84508855999425, -48.00727849400349, 324.87772798513777),
        fpowers=(0.0, 0.0, 0.5),
        domain=(48.5, 180.5),
    ),
    sigma=0.02777896478099617,
    nu=-0.5756035007094116,
    tau=1.7442074097777245,
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
hcmm = BCPEModel(
    mu=FractionalPolynomial(
        intercept=692.5722663851999,
        coefficients=(-641.4702714565977, -309.7469221642888, 72.44232181418973),
        fpowers=(-0.5, 0.0, 0.0),
        domain=(51.0, 195.6),
    ),
    sigma=0.026290598160587913,
    nu=-0.020559517437386845,
    tau=1.567570835467868,
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
hcxx = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-422.2940844051155,
        coefficients=(126.44541423086386, -398.12576972488137, 346.5552682948275),
        fpowers=(0.5, 0.5, 1.0),
        domain=(48.5, 195.6),
    ),
    sigma=0.028047223561619736,
    nu=-0.28912191253351954,
    tau=1.7037869516388158,
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
csfFF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=174.36608687335067,
        coefficients=(0.4198602329356907, -17.850485999978638, 0.10215264932793458),
        fpowers=(-1.0, -0.5, 3.0),
        domain=(-21, 8131),
    ),
    sigma=0.1987401446766343,
    nu=0.42546112691657323,
    tau=1.1913297140133805,
    attrs={"long_name": "CSF Volume", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
csfMM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=190.4495277278507,
        coefficients=(0.44937314250337496, -19.16707542424443, 0.10453302112506725),
        fpowers=(-1.0, -0.5, 3.0),
        domain=(-21.0, 8073.0),
    ),
    sigma=0.20214608231187753,
    nu=0.6562282247254976,
    tau=1.189826112280165,
    attrs={"long_name": "CSF Volume", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
tissFF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-32.40188886990359,
        coefficients=(6.8243795721453555, 1029.4548916362535, -302.4064745323913),
        fpowers=(-0.5, 0.5, 0.5),
        domain=(-21, 8131),
    ),
    sigma=0.07957818731572654,
    nu=-0.4708322009246219,
    tau=2.0429683451866,
    attrs={"long_name": "Brain Tissue", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
tissMM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-85.36920326404015,
        coefficients=(7.81838294854985, 1187.17576187236, -348.1199770487963),
        fpowers=(-0.5, 0.5, 0.5),
        domain=(-21.0, 8073.0),
    ),
    sigma=0.0730139478452033,
    nu=0.6989205425456295,
    tau=1.4925378336133452,
    attrs={"long_name": "Brain Tissue", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
ratioF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=11.082874194353732,
        coefficients=(-4.805927794274573, 3.3948503231064917, -1.021721471224189),
        fpowers=(0.5, 0.5, 0.5),
        domain=(31.8, 332.31),
    ),
    sigma=0.19848713943506638,
    nu=0.05701029564202015,
    tau=1.4670145753100101,
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
ratioM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-10.811516509095394,
        coefficients=(-2.580157458459326, 17.191469945479266, -4.655185756173699),
        fpowers=(0.0, 0.5, 0.5),
        domain=(31.8, 368.55),
    ),
    sigma=0.19355213570641322,
    nu=-0.7371007188630626,
    tau=1.1088596075611707,
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
