#include <cmath>
#include <iostream>
#include <traj_utils/PolyTraj.h>
#include <optimizer/poly_traj_utils.hpp>
#include <mpc_utils/poly_traj_utils.hpp>

int main() {
  traj_utils::PolyTraj msg;
  msg.order = 5;
  msg.duration = {0.5f, 1.25f};
  std::vector<poly_traj::CoefficientMat> ego_coeffs(2);
  for (int piece = 0; piece < 2; ++piece) {
    for (int col = 0; col < 6; ++col) {
      // Asymmetric coefficients exercise all axes and every polynomial power.
      ego_coeffs[piece](0, col) = (piece + 1) * (col + 1) / 16.0;
      ego_coeffs[piece](1, col) = -(piece + 2) * (col + 1) / 32.0;
      ego_coeffs[piece](2, col) = (piece + 3) * (6 - col) / 64.0;
      msg.coef_x.push_back(ego_coeffs[piece](0, col));
      msg.coef_y.push_back(ego_coeffs[piece](1, col));
      msg.coef_z.push_back(ego_coeffs[piece](2, col));
    }
  }
  std::vector<CoefficientMat> mpc_coeffs(2);
  for (int piece = 0; piece < 2; ++piece) {
    for (int col = 0; col < 6; ++col) {
      mpc_coeffs[piece](0, col) = msg.coef_x[piece * 6 + col];
      mpc_coeffs[piece](1, col) = msg.coef_y[piece * 6 + col];
      mpc_coeffs[piece](2, col) = msg.coef_z[piece * 6 + col];
    }
  }
  std::vector<double> durations(msg.duration.begin(), msg.duration.end());
  poly_traj::Trajectory ego(durations, ego_coeffs);
  mpc_utils::Trajectory mpc(durations, mpc_coeffs);
  for (double t : {0.0, 0.1, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 1.0, 1.75}) {
    if ((ego.getPos(t) - mpc.getPos(t)).norm() > 1e-10 ||
        (ego.getVel(t) - mpc.getVel(t)).norm() > 1e-10 ||
        (ego.getAcc(t) - mpc.getAcc(t)).norm() > 1e-10) {
      std::cerr << "FAIL: EGO/MPC trajectory evaluation differs at t=" << t << '\n';
      return 1;
    }
  }
  std::cout << "PASS: EGO PolyTraj coefficients, durations and MPC position/velocity/acceleration agree across piece boundaries\n";
}
