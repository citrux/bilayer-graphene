#define LOWER ( -1 )
#define UPPER ( 1 )

#define Eps ( sqrt(delta12 + p2r + band * sqrt(delta24 + delta32*p2r)) )
#define Eps1 ( sqrt(delta12 + p2r - sqrt(delta24 + delta32*p2r)) )
#define Eps2 ( sqrt(delta12 + p2r + sqrt(delta24 + delta32*p2r)) )
#define rDEps ( fabs(2*Eps/(1 + band * 0.5f * delta32/sqrt(delta24 + delta32 * p2r))) )
#define Vx ( px*(1 + band * 0.5f * delta32 / sqrt(delta24 + delta32 * p2r)) / Eps )
#define Vy ( py*(1 + band * 0.5f * delta32 / sqrt(delta24 + delta32 * p2r)) / Eps )
#define Fx ( Exc + Ex*cos(wx*t) + H*vy )
#define Fy ( Eyc + Ey*cos(wy*t - phi) - H*vx )
#define Value ( vx*Fx+vy*Fy )

#define pi ( 3.14159265f )
#define DELTA(x, width) ( (width) / pi / ((x)*(x) + (width)*(width)) )
#define RNGInit unsigned int x1, y1, z1, w1; x1 = seed[idx]; y1 = \
362436069; z1 = 521288629; w1 = 88675123
#define rand_uniform random_uniform(&x1, &y1, &z1, &w1)

float random_uniform(unsigned int * x1,
                unsigned int * y1,
                unsigned int * z1,
                unsigned int * w1){
    unsigned int t1 = ((*x1)^((*x1)<<11));
    (*x1) = (*y1);
    (*y1) = (*z1);
    (*z1) = (*w1);
    (*w1) = ((*w1)^((*w1)>>19))^(t1^(t1>>8));
    return ((float)(*w1))/(UINT_MAX);
}

__kernel void RNGtest(__global float * mas, __global int * seed){
    int idx = get_global_id(0);
    RNGInit;
    mas[idx] = rand_uniform;
}

__kernel void jobKernel (__global float * dev_average_value_array, \
__global int * num_cols, __global int * up_cols, __global float * dev_params, __global int * \
seed )
{
    // Thread index
    int idx = get_global_id(0);

    // RGN initialization
    RNGInit;
    // Copying parameters in local variables (for speed)
    float Exc = dev_params[0];
    float Eyc = dev_params[1];
    float Ex = dev_params[2];
    float Ey = dev_params[3];
    float H = dev_params[4];
    float wx = dev_params[5];
    float wy = dev_params[6];
    float phi = dev_params[7];
    float wla_max = dev_params[8];
    float wlo_max = dev_params[9];
    float wlr_max = dev_params[10];
    float w0 = dev_params[11];
    float hw = dev_params[12]; 
    float delta1 = dev_params[13];
    float delta2 = dev_params[14];
    float delta3 = dev_params[15];
    float all_time = dev_params[16];
    float dt = dev_params[17];
    float deps = dev_params[18];


    // Prepare variables and work
    float px = 0.0f;
    float py = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float value = 0.0f;
    float temp = 0.0f;
    int numcols = 0;
    int upcols = 0;
    float p = -log(rand_uniform);
    float psi = 2*pi*rand_uniform;
    px = p*cos(psi);
    py = p*sin(psi);
    float t = 0.0f;
    float wsum = 0.0f;
    float r1 = 0.0f;
    float r2 = 0.0f;
    float w[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ps[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float r = - log(rand_uniform);
    float beta = 0.0f;
    float p2r = 0.0f;
    float eps = 0.0f;
    float delta22 = delta2*delta2;
    float delta24 = delta2*delta2*delta2*delta2;
    float delta34 = delta3*delta3*delta3*delta3;
    float delta32 = delta3*delta3;
    float delta12 = delta1*delta1;
    float delta14 = delta12*delta12;
    int i;
    int band = LOWER;

    while( t < all_time ){
        px += Fx*dt;
        py += Fy*dt;
        t += dt;

        p2r = px*px + py*py;
        vx = Vx;
        vy = Vy;
        value += Value*dt;
        eps = Eps;

        float ex = 0;
        float ey = 0;
        if (wx != 0) {
            ex = Ex / wx;        	
        }
        if (wy != 0) {
            ey = Ey / wy;           
        }
        int valley = (idx % 2) * 2 - 1;
        w[4] = (ex*ex + ey*ey + valley*2*ex*ey*sin(phi)) * DELTA(Eps2 - Eps1 - hw, deps);
        

        // Scattering
        if (p2r <  eps * eps - delta12 + delta32/2)
            p2r = 2 * (eps * eps - delta12) + delta32 - p2r;

        w[0] = 0.5f * wla_max * rDEps;
        ps[0] = p2r;

        w[1] = 0.0f;
        ps[1] = 0.0f;
        p2r = 2 * (eps * eps - delta12) + delta32 - p2r;
        if(p2r > 0) {
            w[1] = 0.5f * wla_max * rDEps;
            ps[1] = p2r;
        }

        // w[2] w[3]
        w[2] = 0.0f;
        ps[2] = 0.0f;
        w[3] = 0.0f;
        ps[3] = 0.0f;
        eps -= w0;
        temp = delta24 + delta34 / 4 + delta32 * (eps * eps - delta12);
        if (temp >= 0) {
            p2r = eps * eps - delta12 + delta32/2 - sqrt(temp);
            if(p2r > 0) {
                w[2] = 0.5f * wlo_max * rDEps;
                ps[2] = p2r;
            }

            p2r = 2 * (eps * eps - delta12) + delta32 - p2r;
            w[3] = 0.5f * wlo_max * rDEps;
            ps[3] = p2r;
        }
        eps += w0;

        

        wsum += (w[0] + w[1] + w[2] + w[3] + w[4])*dt;
        if(wsum > r){
            wsum = 0.0f;
            numcols += 1;
            r = rand_uniform;
            psi = atan2(py, px);
            temp = w[0];
            i = 0;
            while(temp < r*(w[0] + w[1] + w[2] + w[3] + w[4])){
                i += 1;
                temp += w[i];
            };
            p = sqrt(ps[i]);
            if (i == 4) {
                if (band == LOWER)
                    upcols+=1;
                band *= -1;
            } else {
                if ( i > 2 ) { eps -= w0; }
                if ( i == 0 || i == 2 ){
                    if (eps * eps > delta12 + delta22) {
                        if (band == LOWER)
                            upcols+=1;
                        band = UPPER;
                    } else {
                        band = LOWER;
                    }
                } else {
                    band = LOWER;
                }
                do {
                    r1 = 2*pi*rand_uniform;
                    r2 = rand_uniform;
                } while(0.5f*(1+cos(r1)) < r2);
                px = p*cos(psi + r1);
                py = p*sin(psi + r1);
                r = - log(rand_uniform);
                };
            };
        };
        value /= t;
        dev_average_value_array [idx] = value;
        num_cols[idx] = numcols;
        up_cols[idx] = upcols;
    }
