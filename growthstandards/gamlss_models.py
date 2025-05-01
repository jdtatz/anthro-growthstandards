from .gamlss_ext import *

hcff = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-274.7848237296913,
        coefficients=(-149.84508855999425, -48.00727849400349, 324.87772798513777),
        fpowers=(0.0, 0.0, 0.5),
        domain=(48.5, 180.5),
        shift=0,
    ),
    sigma=LogLink(-3.5834762072532387),
    nu=-0.5756035007094116,
    tau=LogLink(0.5563002460403006),
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
hcmm = BCPEModel(
    mu=FractionalPolynomial(
        intercept=692.5722663851999,
        coefficients=(-641.4702714565977, -309.7469221642888, 72.44232181418973),
        fpowers=(-0.5, 0.0, 0.0),
        domain=(51.0, 195.6),
        shift=0,
    ),
    sigma=LogLink(-3.6385438880994867),
    nu=-0.020559517437386845,
    tau=LogLink(0.4495271825950411),
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
hcxx = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-422.2940844051155,
        coefficients=(126.44541423086386, -398.12576972488137, 346.5552682948275),
        fpowers=(0.5, 0.5, 1.0),
        domain=(48.5, 195.6),
        shift=0,
    ),
    sigma=LogLink(-3.5738656336728867),
    nu=-0.28912191253351954,
    tau=LogLink(0.5328533922094477),
    attrs={"long_name": "Head Circumference", "units": "cm", "x_long_name": "Height", "x_units": "cm"},
)
csfFF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=174.36608687335067,
        coefficients=(0.4198602329356907, -17.850485999978638, 0.10215264932793458),
        fpowers=(-1.0, -0.5, 3.0),
        domain=(-21, 8131),
        shift=22,
    ),
    sigma=LogLink(-1.615757113209806),
    nu=0.42546112691657323,
    tau=LogLink(0.17507009002314125),
    attrs={"long_name": "CSF Volume", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
csfMM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=190.4495277278507,
        coefficients=(0.44937314250337496, -19.16707542424443, 0.10453302112506725),
        fpowers=(-1.0, -0.5, 3.0),
        domain=(-21.0, 8073.0),
        shift=22,
    ),
    sigma=LogLink(-1.5987646631876429),
    nu=0.6562282247254976,
    tau=LogLink(0.17380717231195147),
    attrs={"long_name": "CSF Volume", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
tissFF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-32.40188886990359,
        coefficients=(6.8243795721453555, 1029.4548916362535, -302.4064745323913),
        fpowers=(-0.5, 0.5, 0.5),
        domain=(-21, 8131),
        shift=22,
    ),
    sigma=LogLink(-2.5310152523814264),
    nu=-0.4708322009246219,
    tau=LogLink(0.7144038214357605),
    attrs={"long_name": "Brain Tissue", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
tissMM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-85.36920326404015,
        coefficients=(7.81838294854985, 1187.17576187236, -348.1199770487963),
        fpowers=(-0.5, 0.5, 0.5),
        domain=(-21.0, 8073.0),
        shift=22,
    ),
    sigma=LogLink(-2.6171047897119526),
    nu=0.6989205425456295,
    tau=LogLink(0.40047791511800596),
    attrs={"long_name": "Brain Tissue", "units": "cm^3", "x_long_name": "Age", "x_units": "days"},
)
## FIXME: broken upstream
_ratioF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=11.082874194353732,
        coefficients=(-4.805927794274573, 3.3948503231064917, -1.021721471224189),
        fpowers=(0.5, 0.5, 0.5),
        domain=(-21, 8131),
        shift=22,
    ),
    sigma=LogLink(-1.6170309696939928),
    nu=0.05701029564202015,
    tau=LogLink(0.3832294345648013),
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
## FIXME: broken upstream
_ratioM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=-10.811516509095394,
        coefficients=(-2.580157458459326, 17.191469945479266, -4.655185756173699),
        fpowers=(0.0, 0.5, 0.5),
        domain=(-21.0, 8073.0),
        shift=22,
    ),
    sigma=LogLink(-1.6422083676281767),
    nu=-0.7371007188630626,
    tau=LogLink(0.10333210663154822),
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
tcF = BCPEModel(
    mu=FractionalPolynomial(
        intercept=7.451604029302721,
        coefficients=(1.9311003202293868, 0.2614952054465591, -0.9333994900219552),
        fpowers=(0.0, 0.0, 1.0),
        domain=(14.0, 8131.0),
        shift=0,
    ),
    sigma=LogLink(-1.7177909182272781),
    nu=0.23164241283324605,
    tau=LogLink(0.45456869049681214),
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
tcM = BCPEModel(
    mu=FractionalPolynomial(
        intercept=7.538670088628258,
        coefficients=(2.2650666623903875, 0.3319242947160484, -1.0650611097770506),
        fpowers=(0.0, 0.0, 1.0),
        domain=(13.0, 8073.0),
        shift=0,
    ),
    sigma=LogLink(-1.7492290015068284),
    nu=0.019907631818795388,
    tau=LogLink(0.29597479412370803),
    attrs={"long_name": "Brain Tissue / CSF Volume Ratio", "x_long_name": "Age", "x_units": "days"},
)
