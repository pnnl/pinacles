from pinacles import ParticlesGrid


def test_classes_exist():
    assert hasattr(ParticlesGrid, "ParticlesBase")
    assert hasattr(ParticlesGrid, "ParticlesSimple")
