double2 P(double2 in){
	return (double2) (-in.x-in.y,in.x*in.y);
}

double2 C(double2 in){
	return (double2) (-in.x*in.x+in.y*in.y-in.x,2*in.x*in.y-in.y);
}

double2 M(double2 pos, double2 pos0){
	return (double2) (pos.x*pos.x-pos.y*pos.y+pos0.x,2*pos.x*pos.y+pos0.y);
}

kernel void schlieren( global uchar* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy)
{
	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;
			
	double delta = 0.5 * (Scale / Resolution);

	double2 pos = (double2) (((double)i / Resolution - 0.5) * Scale + delta - vx, (0.5 - (double)j / Resolution) * Scale - delta - vy);
	
	double2 posdx = pos + (double2) (delta, 0);
	double2 posdy = pos + (double2) (0, delta);	
	double2 pos_dx = pos + (double2) (-delta, 0);
	double2 pos_dy = pos + (double2) (0, -delta);	
	
	for (int k = 0; k < Iterations; k++) {

		if (sign(pos_dx.x) != sign(posdx.x) || sign(pos_dy.x) != sign(posdy.x)) { //VzW
			schlieren[idx] = 1;
			return;
		}
		
		posdx = P(posdx);
		posdy = P(posdy);
		pos_dx = P(pos_dx);
		pos_dy = P(pos_dy);
		
		//posdx = C(posdx);
		//posdy = C(posdy);
		//pos_dx = C(pos_dx);
		//pos_dy = C(pos_dy);
	}
	
	schlieren[idx] = 0;
}

kernel void mandelbrot( global uchar* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy)
{
	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;
			
	double delta = 0.5 * (Scale / Resolution);

	double2 pos = (double2) (((double)i / Resolution - 0.5) * Scale + delta - vx, (0.5 - (double)j / Resolution) * Scale - delta - vy);
	double2 pos0 = pos;

	double2 posdx = pos + (double2) (delta, 0);
	double2 posdy = pos + (double2) (0, delta);	
	double2 pos_dx = pos + (double2) (-delta, 0);
	double2 pos_dy = pos + (double2) (0, -delta);	
	
	double2 posdx0 = posdx;
	double2 pos_dx0 = pos_dx;
	double2 posdy0 = posdy;
	double2 pos_dy0 = pos_dy;

	for (int k = 0; k < Iterations; k++) {

		posdx = M(posdx, posdx0);
		pos_dx = M(pos_dx, pos_dx0);
		posdy = M(posdy, posdy0);
		pos_dy = M(pos_dy, pos_dy0);

		if ((sign(length(posdx)-2.0) != sign(length(pos_dx)-2.0)) || (sign(length(posdy)-2.0) != sign(length(pos_dy)-2.0))){
		
			schlieren[idx] = 1;
			return;

		}

	}

	schlieren[idx] = 0;
}



kernel void scaledown(global uchar * oldbuffer, global uchar * newbuffer, const int oldres){ 

	const int idx = get_global_id(0);
	int newres = oldres / 2;
	
	const int i = idx % newres;
	const int j = idx / newres;

	const int oldi = i * 2;
	const int oldj = j * 2;

	newbuffer[idx] = ((oldbuffer[oldres*oldj + oldi] == 1) | (oldbuffer[oldres*(oldj+1) + oldi] == 1) | (oldbuffer[oldres*(oldj) + oldi + 1] == 1) | (oldbuffer[oldres*(oldj+1) + oldi + 1] == 1))? 1 : 0;

}

kernel void sum(const int res, global int *oldbuffer, global int *newbuffer){ 
	const int idx = get_global_id(0);
	int newres = res / 2;
	
	const int i = (idx % newres);
	const int j = idx / newres;
	const int base_ = i*2 + j*2*res;

	newbuffer[idx] = oldbuffer[base_] + oldbuffer[base_+1] + oldbuffer[base_+res] + oldbuffer[base_+res+1];
}

kernel void firstsum(const int res, global uchar *oldbuffer, global int *newbuffer){ 
	const int idx = get_global_id(0);
	int newres = res / 2;
	
	const int i = (idx % newres);
	const int j = idx / newres;
	const int base_ = i*2 + j*2*res;

	newbuffer[idx] = oldbuffer[base_] + oldbuffer[base_+1] + oldbuffer[base_+res] + oldbuffer[base_+res+1];
}
