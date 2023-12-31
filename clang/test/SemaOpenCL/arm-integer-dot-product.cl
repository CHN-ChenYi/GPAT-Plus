// RUN: %clang_cc1 %s -triple spir-unknown-unknown -finclude-default-header  -fdeclare-opencl-builtins -verify -cl-std=CL1.2 -emit-llvm -o - -cl-ext=-all

void test_negative() {
    uchar4 ua8, ub8;
    char4 sa8, sb8;
    ushort2 ua16, ub16;
    short2 sa16, sb16;
    uint ur;
    int sr;
    ur = arm_dot(ua8, ub8); // expected-error{{implicit declaration of function 'arm_dot' is invalid in OpenCL}}
    sr = arm_dot(sa8, sb8);
    ur = arm_dot_acc(ua8, ub8, ur); // expected-error{{implicit declaration of function 'arm_dot_acc' is invalid in OpenCL}}
    sr = arm_dot_acc(sa8, sb8, sr);
    ur = arm_dot_acc(ua16, ub16, ur);
    sr = arm_dot_acc(sa16, sb16, sr);
    ur = arm_dot_acc_sat(ua8, ub8, ur); // expected-error{{implicit declaration of function 'arm_dot_acc_sat' is invalid in OpenCL}}
    sr = arm_dot_acc_sat(sa8, sb8, sr);
}

