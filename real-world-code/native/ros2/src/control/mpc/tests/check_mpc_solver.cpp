#include <OsqpEigen/OsqpEigen.h>
#include <cmath>
#include <iostream>

int main() {
  Eigen::SparseMatrix<double> hessian(2, 2), constraints(2, 2);
  hessian.insert(0, 0) = hessian.insert(1, 1) = 2.0;
  constraints.setIdentity();
  Eigen::Vector2d gradient(-4.0, 6.0), lower(0.0, -1.0), upper(0.5, 1.0);
  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(false);
  solver.settings()->setAbsoluteTolerance(1e-7);
  solver.settings()->setRelativeTolerance(1e-7);
  solver.data()->setNumberOfVariables(2);
  solver.data()->setNumberOfConstraints(2);
  if (!solver.data()->setHessianMatrix(hessian) ||
      !solver.data()->setGradient(gradient) ||
      !solver.data()->setLinearConstraintsMatrix(constraints) ||
      !solver.data()->setLowerBound(lower) ||
      !solver.data()->setUpperBound(upper) || !solver.initSolver() ||
      solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
    std::cerr << "FAIL: vendored MPC solver could not solve the bounded QP\n";
    return 1;
  }
  const Eigen::VectorXd solution = solver.getSolution();
  if (!solution.allFinite() || (solution - Eigen::Vector2d(0.5, -1.0)).norm() > 1e-5) {
    std::cerr << "FAIL: unexpected solver result " << solution.transpose() << '\n';
    return 1;
  }
  std::cout << "PASS: vendored OSQP/OsqpEigen solved bounded QP: " << solution.transpose() << '\n';
}
