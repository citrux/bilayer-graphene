/*
 *       Electronic transport simulation in bilayer graphene
 *                          version 0.1 
 *                           by citrux
 *
 *                       11th January, 2017
 *
 *             this software is licensed under WTFPL
 */

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>


/*
 * ## Physical (CGS with eV) and mathematical constants
 */
const float e    = 4.8e-10;
const float v_f  = 1e8;
const float v_s  = 2.6e6;
const float c    = 3e10;
const float eV   = 1.6e-12;
const float hbar = 6.56e-16; /* eV */
const float k    = 8.6e-5;   /* eV */
const float pi   = 3.14159265f;


/*
 * ## Needed mathematical functions
 */
inline float sqr(float x) {return x * x;};
inline float dirac_delta(float x, float width) {
    return width / pi / (sqr(x) + sqr(width));
}


enum Mode {
    ONE_BAND,
    TWO_BANDS
};

/*
 * ## Structure for parameters of external conditions and simulation
 */
struct Params {
    /* prefix for output files */
    std::string prefix;
    /* mode: one_band or two_bands  */
    Mode mode;
    /* static fields */
    float E_xc;
    float E_yc;
    float H;
    /* wave */
    float E_x;
    float E_y;
    float omega;
    float phi;
    /* temperature */
    float T;
    /* simulation parameters */
    int n;
    float dt;
    float all_time;
    /* \Delta E \approx \hbar/\tau [undecided] */
    float deps;
};

/*
 * ## Structure for results
 */
struct Data
{
    /* average values */
    /* 
     * maybe add:
     * - energy
     * - band
     */
    float average_v_x;
    float average_v_y;
    float average_power;

    const static int                      types_of_scattering;
    const static std::vector<std::string> scattering_labels;    
    std::vector<int>                      scattering_counts;
    std::vector<float>                   scattering_rates;

    float tau;

    Data () : average_v_x(0),
              average_v_y(0),
              average_power(0),
              tau(0) {
        scattering_counts.assign(types_of_scattering, 0);
        scattering_rates.assign(types_of_scattering, 0);
    }
};

const int Data::types_of_scattering = 3;
const std::vector<std::string> Data::scattering_labels = {"acoustic phonons", "optical phonons", "photons"};

Data & operator+=(Data & lhs, const Data & rhs) {
    lhs.average_v_x += rhs.average_v_x;
    lhs.average_v_y += rhs.average_v_y;
    lhs.average_power += rhs.average_power;
    lhs.tau += rhs.tau;
    for (int i = 0; i < lhs.types_of_scattering; ++i) {
        lhs.scattering_counts[i] += rhs.scattering_counts[i];
        lhs.scattering_rates[i] += rhs.scattering_rates[i];
    }
    return lhs;
}

Data operator*(Data lhs, const Data & rhs) {
    lhs.average_v_x *= rhs.average_v_x;
    lhs.average_v_y *= rhs.average_v_y;
    lhs.average_power *= rhs.average_power;
    lhs.tau *= rhs.tau;
    for (int i = 0; i < lhs.types_of_scattering; ++i) {
        lhs.scattering_counts[i] *= rhs.scattering_counts[i];
        lhs.scattering_rates[i] *= rhs.scattering_rates[i];
    }
    return lhs;
}

Data operator*(Data lhs, int rhs) {
    lhs.average_v_x *= rhs;
    lhs.average_v_y *= rhs;
    lhs.average_power *= rhs;
    lhs.tau *= rhs;
    for (int i = 0; i < lhs.types_of_scattering; ++i) {
        lhs.scattering_counts[i] *= rhs;
        lhs.scattering_rates[i] *= rhs;
    }
    return lhs;
}

Data operator-(Data lhs, const Data & rhs) {
    lhs.average_v_x -= rhs.average_v_x;
    lhs.average_v_y -= rhs.average_v_y;
    lhs.average_power -= rhs.average_power;
    lhs.tau -= rhs.tau;
    for (int i = 0; i < lhs.types_of_scattering; ++i) {
        lhs.scattering_counts[i] -= rhs.scattering_counts[i];
        lhs.scattering_rates[i] -= rhs.scattering_rates[i];
    }
    return lhs;
}

Data operator/(Data lhs, int rhs) {
    lhs.average_v_x /= rhs;
    lhs.average_v_y /= rhs;
    lhs.average_power /= rhs;
    lhs.tau /= rhs;
    for (int i = 0; i < lhs.types_of_scattering; ++i) {
        lhs.scattering_counts[i] /= rhs;
        lhs.scattering_rates[i] /= rhs;
    }
    return lhs;
}

Data sqrt(Data d) {
    d.average_v_x = sqrt(d.average_v_x);
    d.average_v_y = sqrt(d.average_v_y);
    d.average_power = sqrt(d.average_power);
    d.tau = sqrt(d.tau);
    for (int i = 0; i < d.types_of_scattering; ++i) {
        d.scattering_counts[i] = sqrt(d.scattering_counts[i]);
        d.scattering_rates[i] = sqrt(d.scattering_rates[i]);
    }
    return d;
}

/*
 * ## Random number generator
 *
 * Simple and fast xorshift for use in simulation
 *
 */
struct Rng {
    unsigned int x, y, z, w;

    Rng (unsigned int seed) : x(seed),
                              y(362436069),
                              z(521288629),
                              w(88675123) {};
    
    float uniform () {
        unsigned int t = x ^ (x << 1);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t ^ (t >> 8);
        return ((float) w) / UINT_MAX;
    }
};

namespace bigraphene {
    enum Band {
        LOWER = -1,
        UPPER = 1
    };

    /*
     * It's very comfortable to define all of material-specific expressions in one place,
     * but I think it also can be done without macros.
     */
    #define rDEps ( fabs(2 * eps / (1 + band * 0.5f * (gamma2+4*delta2)/sqrt(gamma4/4 + (gamma2+4*delta2) * p2))) )
    
    const float delta = 1e-3;
    const float gamma = 0.35;
    
    const float optical_phonon_energy = 0.196;
    const float rho = 2 * 7.7e-8;
    const float Dak = 18;
    const float Dopt = 1.4e9;

    /*
     * TODO: [x] сделать так, чтобы код компилировался 
     *       [ ] использовать подходящие имена переменных в промежуточных расчётах
     *       [ ] уйти от макросов везде, где это возможно
     *       [ ] добавить вычисление усреднённой функции распределения (это сильно сложнее)
     *       [ ] уменьшить кол-во кода в jobKernel
     */

    void one_particle_simulation (const int idx,
                                  const int seed,
                                  const Params & params,
                                  Data & data) {
        auto rng = Rng(seed);

        int valley = (idx % 2) * 2 - 1;

        const float field_dimensionless_factor = e * v_f * params.dt / eV;
        const float E_xc = params.E_xc * field_dimensionless_factor;
        const float E_yc = params.E_yc * field_dimensionless_factor;
        const float E_x  = params.E_x  * field_dimensionless_factor;
        const float E_y  = params.E_y  * field_dimensionless_factor;
        const float H    = params.H * v_f / c * field_dimensionless_factor;
        
        const float omega = params.omega * params.dt;
        const float phi   = params.phi;
        const float photon_energy = hbar * params.omega;
        
        const int all_time = params.all_time / params.dt;
        const float deps = params.deps;
        const float T = params.T;

        const float wla_max = sqr(k * T * Dak) * eV / (2 * pow(hbar, 3) *
                                                        rho * sqr(v_s * v_f));
        const float wlo_max = k * T * sqr(Dopt) * eV / (4 * hbar *
                                                         optical_phonon_energy *
                                                         rho * sqr(v_f));

        // Prepare variables and work
        int numcols = 0;
        float temp = 0;
        float w[5];
        float ps[5];
        float sum_w[data.types_of_scattering];
    	for (int i = 0; i < data.types_of_scattering; ++i) {
                sum_w[i] = 0;
            }
        float p = 0; // -log(rng.uniform());
        float psi = 0; // 2 * pi * rng.uniform();
        float p_x = p * cos(psi);
        float p_y = p * sin(psi);
        float wsum = 0.0f;
        float r1 = 0.0f;
        float r2 = 0.0f;
        float r = -log(rng.uniform());
        Band band = LOWER;

        const float E_x2 = sqr(E_x);
        const float E_y2 = sqr(E_y);
        const float E_xE_y = E_x * E_y;
        const float delta2 = sqr(delta);
        const float gamma2 = sqr(gamma);
        const float gamma4 = sqr(gamma2);

        float sum_v_x = 0;
        float sum_v_y = 0;
        float sum_power = 0;

        for (int t = 0; t < all_time; ++t) {
            float p_x2 = sqr(p_x);
            float p_y2 = sqr(p_y);
            float p2 = p_x2 + p_y2;

            float eps_1 = sqrt(delta2 + gamma2 / 2 + p2 -
                                sqrt(gamma4 / 4 + (gamma2 + 4 * delta2) * p2));
            float eps_2 = sqrt(delta2 + gamma2 / 2 + p2 +
                                sqrt(gamma4 / 4 + (gamma2 + 4 * delta2) * p2));
            float eps = (band == LOWER) ? eps_1 : eps_2;

            float v_x = p_x / eps *
                         (1 + band * 0.5f * (gamma2 + 4 * delta2) /
                              sqrt(gamma4 / 4 + (gamma2 + 4 * delta2) * p2));
            float v_y = p_y / eps *
                         (1 + band * 0.5f * (gamma2 + 4 * delta2) /
                              sqrt(gamma4 / 4 + (gamma2 + 4 * delta2) * p2));

            sum_v_x += v_x;
            sum_v_y += v_y;

            const float f_x = E_xc + E_x * cos(omega * t) + v_y * H;
            const float f_y = E_yc + E_y * cos(omega * t + phi) - v_x * H;
            
            float power = f_x * v_x + f_y * v_y;
            sum_power += power;

            float e_1pd = eps_1 + delta;
            float e_1md = eps_1 - delta;
            float e_2pd = eps_2 + delta;
            float e_2md = eps_2 - delta;
            
            float e_1pd2 = sqr(e_1pd);
            float e_1md2 = sqr(e_1md);
            float e_2pd2 = sqr(e_2pd);
            float e_2md2 = sqr(e_2md);

            float lambda = (e_1pd2-p2)*(e_2pd2-p2)/gamma2;
            float A_1 = e_1md * e_1pd + lambda;
            float A_2 = e_1md * (e_2pd + lambda / e_2md);
            float A_12 = sqr(A_1);
            float A_22 = sqr(A_2);

            float B = (p2 - 4 * delta2) / e_1pd / e_2md / e_2pd;
            float B2 = sqr(B);
            float theta = atan2(p_y, p_x);

            w[4] = 0;
            if (params.mode == TWO_BANDS) {
                w[4] = (
                    gamma4 / omega / omega  * (
                        A_12*(E_x2+E_y2-2*valley*E_xE_y*sin(phi))+
                        A_22*(E_x2+E_y2+2*valley*E_xE_y*sin(phi))+
                        2*A_1*A_2*((E_x2-E_y2) * cos(2 * theta) +
                                 2 * E_xE_y * cos(phi) * sin(2 * theta))
                    ) / (
                        (gamma2*p2*B2 * (e_1pd2 + p2) + (p2 * B2 + 1) * sqr(e_1pd2-p2))*
                        (gamma2*(e_2pd2 + p2) + (p2 / e_2md2 + 1) * sqr(e_2pd2-p2))
                    )
                ) * pi / 8 / hbar * dirac_delta(eps_2 - eps_1 - photon_energy, deps);
            }
            // if (p2 == 0) {
            //     printf("%e %e %e %e\n", w[4], eps_2 - eps_1 - photon_energy, deps, dirac_delta(eps_2 - eps_1 - photon_energy, deps));
            // }
            // Scattering
            // w[0] and w[1] are related to acoustic scattering
            if (p2 <  sqr(eps) - (delta2+gamma2/2) + (gamma2 + 4*delta2)/2)
                p2 = 2 * (sqr(eps) - (delta2+gamma2/2)) + (gamma2 + 4*delta2) - p2;

            w[0] = 0.5f * wla_max * rDEps;
            ps[0] = p2;

            w[1] = 0.0f;
            ps[1] = 0.0f;
            p2 = 2 * (sqr(eps) - (delta2+gamma2/2)) + (gamma2 + 4*delta2) - p2;
            if(p2 > 0) {
                w[1] = 0.5f * wla_max * rDEps;
                ps[1] = p2;
            }

            // w[2] and w[3] are related to scattering on optical phonons
            w[2] = 0.0f;
            ps[2] = 0.0f;
            w[3] = 0.0f;
            ps[3] = 0.0f;
            eps -= optical_phonon_energy;
            temp = gamma4/4 + sqr(gamma2 + 4*delta2) / 4 + (gamma2 + 4*delta2) * (eps * eps - delta2 - gamma2/2);
            if (temp >= 0) {
                p2 = eps * eps - (delta2+gamma2/2) + (gamma2 + 4*delta2)/2 - sqrt(temp);
                if(p2 > 0) {
                    w[2] = 0.5f * wlo_max * rDEps;
                    ps[2] = p2;
                }

                p2 = 2 * (eps * eps - (delta2+gamma2/2)) + (gamma2 + 4*delta2) - p2;
                w[3] = 0.5f * wlo_max * rDEps;
                ps[3] = p2;
            }
            eps += optical_phonon_energy;

            if (params.mode == ONE_BAND && eps * eps > delta2 + gamma2) {
                w[0] = 0;
                w[2] = 0;
            }

            sum_w[0] += w[0] + w[1];
            sum_w[1] += w[2] + w[3];
            sum_w[2] += w[4]; 
            wsum += (w[0] + w[1] + w[2] + w[3] + w[4]) * params.dt;
            if(wsum > r) {
                wsum = 0.0f;
                ++numcols;
                r = rng.uniform();
                psi = atan2(p_y, p_x);
                temp = w[0];
                int i = 0;
                while(temp < r*(w[0] + w[1] + w[2] + w[3] + w[4])){
                    temp += w[++i];
                };
                p = sqrt(ps[i]);
                if (i == 4) {
                    band = (band == LOWER) ? UPPER : LOWER;
                    ++data.scattering_counts[2];
                } else {
                    if (i < 2) {
                        ++data.scattering_counts[0];
                    }
                    if ( i > 1 ) {
                        eps -= optical_phonon_energy;
                        ++data.scattering_counts[1];
                    }
                    if ( i == 0 || i == 2 ){
                        if (eps * eps > delta2 + gamma2) {
                            band = UPPER;
                        } else {
                            band = LOWER;
                        }
                    } else {
                        band = LOWER;
                    }
                    
                    do {
                        r1 = 2 * pi * (rng.uniform() - 0.5);
                        r2 = rng.uniform();
                    } while(r2 < 0.5f * (1 + cos(r1)));
                    p_x = p * cos(psi + r1);
                    p_y = p * sin(psi + r1);
                    r = - log(rng.uniform());
                }
            }

            p_x += f_x;
            p_y += f_y;
        }
    
        data.average_v_x   = sum_v_x   / all_time;
        data.average_v_y   = sum_v_y   / all_time;
        data.average_power = sum_power / all_time;

        for (int i = 0; i < data.types_of_scattering; ++i) {
            data.scattering_rates[i] = sum_w[i] / all_time;
        }

        data.tau = params.all_time / (numcols + 1);
    }
}

void puts_center(const char * str, char fillchar) {
    int len = strlen(str);
    int width = 80;
    int pad_left = (width - len) / 2;
    int pad_right = width - len - pad_left;
    for (int i = 0; i < pad_left; ++i) { putchar(fillchar); }
    printf(str);
    for (int i = 0; i < pad_right; ++i) { putchar(fillchar); }
    putchar('\n');
}

int main(int argc, char const *argv[])
{
    Params params;
    puts_center("Electronic transport simulation in bilayer graphene", ' ');
    puts_center(" [ v 0.1 ] ", '-');
    puts_center("by citrux", ' ');
    puts("");
    puts("Waiting for input...");
    std::cin >> params.prefix;
    std::string mode;
    std::cin >> mode;
    if (mode == "one_band") {
        params.mode = ONE_BAND;
    }
    else {
        params.mode = TWO_BANDS;
    }
    std::cin >> params.E_xc;
    std::cin >> params.E_yc;
    std::cin >> params.H;
    std::cin >> params.E_x;
    std::cin >> params.E_y;
    std::cin >> params.omega;
    std::cin >> params.phi;
    std::cin >> params.T;
    std::cin >> params.n;
    std::cin >> params.dt;
    std::cin >> params.all_time;
    std::string type;
    std::cin >> type;
    std::cin >> params.deps;
    if (type == "tau") {
        params.deps = hbar / params.deps;
    }

    puts("Input data was read successfully!");
    printf("prefix   = %s\n", params.prefix.c_str());
    printf("mode     = %s\n", (params.mode == ONE_BAND) ? "one band" : "two bands");
    printf("E_xc     = %e\n", params.E_xc);
    printf("E_yc     = %e\n", params.E_yc);
    printf("H        = %e\n", params.H);
    printf("E_x      = %e\n", params.E_x);
    printf("E_y      = %e\n", params.E_y);
    printf("omega    = %e\n", params.omega);
    printf("phi      = %f\n", params.phi);
    printf("T        = %f\n", params.T);
    printf("n        = %d\n", params.n);
    printf("dt       = %e\n", params.dt);
    printf("all_time = %e\n", params.all_time);
    printf("deps     = %e\n", params.deps);
    
    srand(time(nullptr));
    Data *data = new Data[params.n];
    #pragma omp parallel for
    for (int i = 0; i < params.n; ++i) {
        bigraphene::one_particle_simulation(i, rand(), params, data[i]);
    }
    Data sum, mean, sum_sqr, stdev;
    for (int i = 0; i < params.n; ++i) {
        sum += data[i];
        sum_sqr += data[i] * data[i];
    }
    mean  = sum / params.n;
    stdev = sqrt(sum_sqr - mean * sum) / params.n;
    printf("\nResults:\n");
    printf("vx    = %e +/- %e v_f\n", mean.average_v_x, stdev.average_v_x);
    printf("vy    = %e +/- %e v_f\n", mean.average_v_y, stdev.average_v_y);
    printf("power = %e +/- %e eV/s\n", mean.average_power, stdev.average_power);
    printf("tau   = %e +/- %e s\n", mean.tau, stdev.tau);
    printf("\nScattering:\n");
    for (int i = 0; i < mean.types_of_scattering; ++i) {
        printf("%s = %d +/- %d\n", mean.scattering_labels[i].c_str(), mean.scattering_counts[i], stdev.scattering_counts[i]);
    }
    printf("\nScattering rates:\n");
    for (int i = 0; i < mean.types_of_scattering; ++i) {
        printf("%s = %e +/- %e\n", mean.scattering_labels[i].c_str(), mean.scattering_rates[i], stdev.scattering_rates[i]);
    }
    return 0;
}
