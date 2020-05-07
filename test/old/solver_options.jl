
# Make sure all solver options can be created without specifying float type
@test_nowarn ilqr = iLQRSolverOptions()
@test_nowarn al = AugmentedLagrangianSolverOptions()
@test_nowarn altro = ALTROSolverOptions()
@test_nowarn pn = ProjectedNewtonSolverOptions()
