from pinacles import parameters


def test_cp():
    # Check that specific heat of dry air and inverse specific heat are consistent
    assert parameters.CPD == 1.0 / parameters.ICPD

    # Check at all specific heats are posistive
    assert parameters.CL > 0
    assert parameters.CI > 0
    return


def test_Rs():
    # Gas constants should always be positive
    assert parameters.RD > 0
    assert parameters.RV > 0

    assert parameters.RD == parameters.KAPPA * parameters.CPD
    return


def test_G():
    # We assume that the gravitational accleration is positive down
    assert parameters.G >= 0
    return


def test_kappa():
    assert parameters.KAPPA == parameters.RD / parameters.CPD
    return


def test_epsv():
    assert parameters.RD / parameters.RV
    assert parameters.EPSV == 1.0 / parameters.EPSVI
    return


def test_omega():
    # The rotational frequency of the earth should be positive
    assert parameters.OMEGA >= 0.0
    return


def test_P00():
    # The reference pressure should always be positive
    assert parameters.P00 > 0
    return
