/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
mininum image convention
------------------------------------------------------------------------------*/


#define LDG(a, n) __ldg(a + n) // sm_35
//#define LDG(a, n) a[n] // sm_20


static __device__ void dev_apply_mic
(
    int triclinic,
    const double* __restrict__ h, double &x12, double &y12, double &z12
)
{
    if (triclinic == 0) // orthogonal box
    {
        if      (x12 < - LDG(h,0) * 0.5) {x12 += LDG(h,0);}
        else if (x12 > + LDG(h,0) * 0.5) {x12 -= LDG(h,0);}
        if      (y12 < - LDG(h,1) * 0.5) {y12 += LDG(h,1);}
        else if (y12 > + LDG(h,1) * 0.5) {y12 -= LDG(h,1);}
        if      (z12 < - LDG(h,2) * 0.5) {z12 += LDG(h,2);}
        else if (z12 > + LDG(h,2) * 0.5) {z12 -= LDG(h,2);}
    }
    else // triclinic box
    {
        double sx12 = LDG(h,9)  * x12 + LDG(h,10) * y12 + LDG(h,11) * z12;
        double sy12 = LDG(h,12) * x12 + LDG(h,13) * y12 + LDG(h,14) * z12;
        double sz12 = LDG(h,15) * x12 + LDG(h,16) * y12 + LDG(h,17) * z12;
        sx12 -= nearbyint(sx12);
        sy12 -= nearbyint(sy12);
        sz12 -= nearbyint(sz12);
        x12 = LDG(h,0) * sx12 + LDG(h,1) * sy12 + LDG(h,2) * sz12;
        y12 = LDG(h,3) * sx12 + LDG(h,4) * sy12 + LDG(h,5) * sz12;
        z12 = LDG(h,6) * sx12 + LDG(h,7) * sy12 + LDG(h,8) * sz12;
    }
}


