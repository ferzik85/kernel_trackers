
#include "MOSSETracker.h"
#include <algorithm>
#include <vector>
#include "imResample.cpp"
#include "gradient.h"
//#include <math.h>

#define PI 3.14159265f
const float eps = std::numeric_limits<float>::epsilon();

namespace mosse {

	mosse_tracker::mosse_tracker(bool _mosse, float _padding, float _output_sigma_factor,
		float _scale_sigma_factor, float _lambda1, float _lambda2,
		float _learning_rate, int _number_of_scales,
		float _scale_step, int _scale_model_max_area, bool _use_scale_4_translation_estimate)
	{
		if (_padding < 0 && _padding > 2) padding = 2.0f;
		else padding = _padding;

		if (_output_sigma_factor < 0.01f && _output_sigma_factor > 0.5f) output_sigma_factor = 0.1f;
		else output_sigma_factor = _output_sigma_factor;

		if (_scale_sigma_factor < 0.01f && _scale_sigma_factor > 1.0f) scale_sigma_factor = 0.25f;
		else scale_sigma_factor = _scale_sigma_factor;

		if (_lambda1 < 0.0f && _lambda1 > 1.0f) lambda1 = 0.0001f;
		else lambda1 = _lambda1;

		if (_lambda2 < 0.0f && _lambda2 > 100.0f) lambda2 = 20.f;
		else lambda2= _lambda2;

		if (_learning_rate < 0.0f && _learning_rate > 1.0f) learning_rate = 0.05f; // ��������� ���� �������� ��� �������� ��������
		else learning_rate = _learning_rate;

		if (_number_of_scales < 0 && _number_of_scales > 50) nScales = 33;
		else nScales = _number_of_scales;

		if (_scale_step< 1 && _scale_step > 2) scale_step = 1.02f;
		else scale_step = _scale_step;

		if (_scale_model_max_area < 256 && _scale_model_max_area > 1024) scale_model_max_area = 512;
		else scale_model_max_area = _scale_model_max_area;

		if (_mosse == true) {
			dout = 1; cell_size = 1; learning_rate = 0.05f; lambda2 = 20.f;// gray
		}
		else {
			dout = 31; cell_size = 4; learning_rate = 0.015f; lambda2 = 25.f;  // hog - all-zeros channel features are not considered
		}

		comp_learning_rate = 1 - learning_rate;
		dscale = 31; 	
		xt = (float**)fftwf_malloc(sizeof(float*) * dout);
		xtf = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * dout);
		kf = (float**)fftwf_malloc(sizeof(float*) * dout);
		num = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * dout);
		model_wf = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * dout);		
		use_scale_4_translation_estimate = _use_scale_4_translation_estimate;
		scale_window = new float[nScales]; scaleFactors = new float[nScales];
		for (int ss = 0; ss < nScales; ss++)
			scaleFactors[ss] = pow(scale_step, ceil(float(nScales) / 2) - (ss + 1));
		currentScaleFactor = 1.0f; imw = imh = 0;
	}

	mosse_tracker::~mosse_tracker()
	{
		for (int j = 0; j < dout; j++)
			fftwf_destroy_plan(pxtf[j]);
		fftwf_free(pxtf);
		for (int i = 0; i < sizess; i++)
			fftwf_destroy_plan(pxsf[i]);
		fftwf_free(pxsf);
		fftwf_destroy_plan(prespt);
		fftwf_destroy_plan(presps);
		fftwf_destroy_plan(pyf);
		fftwf_destroy_plan(pysf);
		fftwf_free(yf); fftwf_free(ysf);
		for (int i = 0; i < dout; i++) fftwf_free(xt[i]);
		fftwf_free(xt);
		for (int i = 0; i < dout; i++) fftwf_free(xtf[i]);
		fftwf_free(xtf);
		for (int i = 0; i < dout; i++) fftwf_free(model_wf[i]);
		fftwf_free(model_wf);		
		fftwf_free(rt); fftwf_free(respt);
		fftwf_free(rts); fftwf_free(resps);
		for (int i = 0; i < sizess; i++) fftwf_free(xs[i]);
		fftwf_free(xs);
		for (int i = 0; i < sizess; i++) fftwf_free(xsf[i]);
		fftwf_free(xsf);
		for (int i = 0; i < sizess; i++) fftwf_free(sf_num[i]);
		fftwf_free(sf_num);
		fftwf_free(sf_den);
		delete[] scaleFactors;
		delete[] scale_window;
		delete[] cos_window;
		for (int i = 0; i < dout; i++) fftwf_free(kf[i]);
		fftwf_free(kf);
		for (int i = 0; i < dout; i++) fftwf_free(num[i]);
		fftwf_free(num);
		fftwf_cleanup();
	}

	void mosse_tracker::make_translation_hann_window(int n, float* window)
	{
		for (int i = 0; i < n; i++) {
			window[i] = 0.5f * (1.0f - cos(2.0f * PI*i / (n - 1)));
		}
	}

	void mosse_tracker::make_scale_hann_window(int n, float *window)
	{
		if (n > nScales)
			for (int i = 1; i < n; i++)
				window[i - 1] = 0.5f * (1.f - cos(2.f * PI*i / (n - 1)));
		else
			for (int i = 0; i < n; i++) {
				window[i] = 0.5f * (1.f - cos(2.f * PI*i / (n - 1)));
			}
	}

	void mosse_tracker::make_scale_cosine_mask()
	{
		if (nScales % 2 == 0)
			make_scale_hann_window(nScales + 1, scale_window);
		else
			make_scale_hann_window(nScales, scale_window);
	}

	void mosse_tracker::make_translation_cosine_mask()
	{
		float *w = new float[csz[1]];
		float *h = new float[csz[0]];

		make_translation_hann_window(csz[1], w);
		make_translation_hann_window(csz[0], h);

		cos_window = new float[csz[1] * csz[0]];

		for (int i = 0; i < csz[0]; i++)
			for (int j = 0; j < csz[1]; j++) {
				cos_window[j + i*csz[1]] = h[i] * w[j];
			}

		delete[] w; delete[] h;
	}

	bool mosse_tracker::initializeTargetModel(int c_x, int c_y, int t_w, int t_h, int _imw, int _imh, unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB)
	{
		bool goodinit = true;
		if (c_x < 0 || c_x > _imw - 1) { c_x = _imw / 2; goodinit = false; }
		if (c_y < 0 || c_y > _imh - 1) { c_y = _imh / 2; goodinit = false; }
		if (t_w < 4) t_w = 4; 
		if (t_h < 4) t_h = 4;

		imw = _imw; imh = _imh; ix = c_x; iy = c_y; iw = t_w; ih = t_h; score = 1;
		base_target_sz[0] = ih;  base_target_sz[1] = iw;
		target_size[0] = base_target_sz[0]; target_size[1] = base_target_sz[1];
		sz[0] = (int)floor(base_target_sz[0] * (1.0 + padding));
		sz[1] = (int)floor(base_target_sz[1] * (1.0 + padding));

		// set offsets
		offset[0][0] = -base_target_sz[0]; offset[0][1] = 0;
		offset[1][0] = 0; offset[1][1] = -base_target_sz[1];
		offset[2][0] = base_target_sz[0]; offset[2][1] = 0;
		offset[3][0] = 0; offset[3][1] = base_target_sz[1];

		// desired translation filter output(gaussian shaped), bandwidth proportional to target size
		float output_sigma = sqrt(float(base_target_sz[0] * base_target_sz[1])) * output_sigma_factor / cell_size;
		float output_sigma2 = output_sigma * output_sigma;

		csz[0] = (int) floor( float(sz[0]) / cell_size);
		csz[1] = (int) floor( float(sz[1]) / cell_size);
		prodsz = csz[0] * csz[1];
		prodszhalf = csz[0] * (csz[1] / 2 + 1);
		float *y = (float*)fftwf_malloc(sizeof(float) *prodsz);
		memset(y, 0, sizeof(float) *prodsz);
		yf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * prodszhalf);
		memset(yf, 0, sizeof(fftwf_complex) * prodszhalf);
	
		pyf = fftwf_plan_dft_r2c_2d(csz[0], csz[1], y, yf, FFTW_ESTIMATE);

		int dy = 1 - (int)floor(float(csz[0]) / 2); // <0
		int dx = 1 - (int)floor(float(csz[1]) / 2); // <0
		int in = 0; int in1 = 0;
		
		// circular shift of y in 2D
		for (int rs = dy; rs < csz[0] + dy; rs++)
			for (int cs = dx; cs < csz[1] + dx; cs++)
			{     
				if (cs < 0) { // �����
					if (rs < 0)
						in = (csz[1] + cs) + csz[1] * (csz[0] + rs);  // �����
					else
						in = (csz[1] + cs) + csz[1] * (rs - dy + dy); // ������ ��� ����� ����
				}
				else // ������ ��� ����� ����
					if (rs < 0)
						in = (cs - dx + dx) + csz[1] * (csz[0] + rs);  // �����
					else
						in = (cs - dx + dx) + csz[1] * (rs - dy + dy); // ������ ��� ����� ����			
				y[in] = exp(-0.5f * (((rs * rs + cs * cs) / output_sigma2)));							
			}

		fftwf_execute(pyf);
		fftwf_free(y);

		// desired scale filter output(gaussian shaped), bandwidth proportional to number of scales
		float scale_sigma = nScales / sqrt(33.0f) * scale_sigma_factor;
		int *ss = new int[nScales];
		for (int i = 1; i < nScales + 1; i++)
			ss[i - 1] = i - (int)ceil(float(nScales) / 2);

		ysf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (nScales / 2 + 1));
		float *ys = (float*)fftwf_malloc(sizeof(float) * nScales);
		pysf = fftwf_plan_dft_r2c_1d(nScales, ys, ysf, FFTW_ESTIMATE);

		float scale_sigma2 = scale_sigma*scale_sigma;
		for (int i = 0; i < nScales; i++)
			ys[i] = exp(-0.5f * (ss[i] * ss[i]) / scale_sigma2);
		delete[] ss;

		fftwf_execute(pysf);
		fftwf_free(ys);

		// store pre - computed translation filter cosine window
		make_translation_cosine_mask();

		// Create the cosine mask used for the scale filtering.
		make_scale_cosine_mask();

		// compute the resize dimensions used for feature extraction in the scale estimation
		float scale_model_factor = 1.0f;
		int square = iw*ih;
		if (square > scale_model_max_area)
			scale_model_factor = sqrt(float(scale_model_max_area) / square);
		scale_model_sz[0] = (int)floor(ih * scale_model_factor);
		scale_model_sz[1] = (int)floor(iw * scale_model_factor);

		sizess = dscale * (scale_model_sz[0] / 4) * (scale_model_sz[1] / 4);

		xs = (float**)fftwf_malloc(sizeof(float*)*sizess);

		for (int i = 0; i < sizess; i++)
			xs[i] = (float*)fftwf_malloc(sizeof(float) * nScales);

		xsf = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*)*sizess);
		for (int i = 0; i < sizess; i++)
			xsf[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (nScales / 2 + 1));

		pxsf = (fftwf_plan*)fftwf_malloc(sizeof(fftwf_plan) * sizess);
		for (int i = 0; i < sizess; i++)
			pxsf[i] = fftwf_plan_dft_r2c_1d(nScales, xs[i], xsf[i], FFTW_MEASURE);//FFTW_ESTIMATE);

		sf_num = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*)*sizess);
		for (int i = 0; i < sizess; i++)
			sf_num[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (nScales / 2 + 1));

		rts = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (nScales / 2 + 1));

		sf_den = (float*)fftwf_malloc(sizeof(float) * nScales);

		resps = (float*)fftwf_malloc(sizeof(float) * nScales);

		presps = fftwf_plan_dft_c2r_1d(nScales, rts, resps, FFTW_MEASURE);//FFTW_ESTIMATE);

		// find maximum and minimum scales
		min_scale_factor = powf(scale_step, ceil(logf(std::max(5.0f / sz[0], 5.0f / sz[1])) / logf(scale_step)));
		max_scale_factor = powf(scale_step, floorf(logf(std::min(float(imh) / base_target_sz[0], float(imw) / base_target_sz[1])) / logf(scale_step)));

		// allocate memory
		for (int i = 0; i < dout; i++)
			xt[i] = (float*)fftwf_malloc(sizeof(float) * csz[1] * csz[0]);

		for (int i = 0; i < dout; i++)
			kf[i] = (float*)fftwf_malloc(sizeof(float) * prodszhalf);

		for (int i = 0; i < dout; i++)
			num[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * prodszhalf);

		for (int i = 0; i < dout; i++)
			xtf[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * prodszhalf);

		for (int i = 0; i < dout; i++)
			model_wf[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * prodszhalf);

		pxtf = (fftwf_plan*)fftwf_malloc(sizeof(fftwf_plan) * dout);
		for (int j = 0; j < dout; j++)
			pxtf[j] = fftwf_plan_dft_r2c_2d(csz[0], csz[1], xt[j], xtf[j], FFTW_MEASURE); // FFTW_ESTIMATE);

		rt = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * prodszhalf);

		respt = (float*)fftwf_malloc(sizeof(float) * prodsz);

		prespt = fftwf_plan_dft_c2r_2d(csz[0], csz[1], rt, respt, FFTW_MEASURE); // FFTW_ESTIMATE);
		
		// extract the training sample feature map for the translation filter	
		extract_training_sample_info(dataYorR, dataG, dataB, true);

		return goodinit;
	}

	void mosse_tracker::get_translation_feature_map(float *In, float **Out, int h, int w, int din)
	{
		int n = h*w; int n2 = n * 2; int r, c;

		if (dout == 1) { // use only gray color
			// compute gray features and save in row-major order	
			if (din == 3) {
				float mean = 0.0f;
				for (int i = 0; i < h; i++)
					for (int j = 0; j < w; j++) {
						r = j + i*w; c = i + j*h;
						Out[0][r] = (0.2989f*In[c] + 0.5870f*In[c + n] + 0.1140f*In[c + n2])/255;
						mean += Out[0][r];
					}
				mean /= h*w;
				for (int i = 0; i < h; i++)
					for (int j = 0; j < w; j++) {
						r = j + i*w;
						Out[0][r] = (Out[0][r] - mean) * cos_window[r];
					}
			}
			else
				if (din == 1) {
					float mean = 0.0f;
					// compute mean and normalize values
					for (int i = 0; i < h; i++)
						for (int j = 0; j < w; j++) {
							r = j + i*w;
							Out[0][r] = In[i + j*h] / 255;
							mean += Out[0][r];
						}
					mean /= h*w;
					// substract mean and apply cousine
					for (int i = 0; i < h; i++)
						for (int j = 0; j < w; j++) {
							r = j + i*w;
							Out[0][r] = (Out[0][r] - mean) * cos_window[r];
						}
				}
		}
		else { // use only Hog

			//compute fhog features in col-major order	
			float *normI = (float*)fftwf_malloc(sizeof(float) * n);	
			if (din == 3) {
				for (int i = 0; i < h; i++)
					for (int j = 0; j < w; j++) {
						 c = i + j*h;
						 normI[c] = (0.2989f*In[c] + 0.5870f*In[c + n] + 0.1140f*In[c + n2]) / 255;
					}
			}
			else
				if (din == 1) {
					for (int i = 0; i < n; i++) {
						normI[i] = In[i] / 255;
					}
				}

			float *M = (float*)fftwf_malloc(sizeof(float) * n);
			float *O = (float*)fftwf_malloc(sizeof(float) * n);
			gradientMagnitude(normI, M, O, h, w, 1, true); //gradientMagnitude(normI, M, O, h, w, din, true);
			int binSize = cell_size; int nOrients = 9; int softBin = -1; float clip = 0.2f;
			int hb = h / binSize; int wb = w / binSize; int nChns = nOrients * 3 + 5; int nb = hb*wb;
			float *H = (float*)fftwf_malloc(sizeof(float) * hb*wb*nChns);
			memset(H, 0, sizeof(float)*hb*wb*nChns); //to be sure it is set to 0
			fhog(M, O, H, h, w, binSize, nOrients, softBin, clip);
			
			// save hog features in row-major order
			for (int c = 0; c < dout; c++)
				for (int i = 0; i < hb; i++)
					for (int j = 0; j < wb; j++) {
						r = j + i*wb;
						Out[c][r] = H[i + j*hb + nb*c] * cos_window[r];
					}

			fftwf_free(H);	fftwf_free(M);	fftwf_free(O); fftwf_free(normI);
		}
	}

	void mosse_tracker::get_scale_feature_map(float *In, int h, int w, int din, float **scale_sample, int s)
	{
		// compute fhog features in col-major order
		int n = h*w;
		//float *M = new float[n]; 
		//float *O = new float[n];
		float *M = (float*)fftwf_malloc(sizeof(float)*n);
		float *O = (float*)fftwf_malloc(sizeof(float)*n);
		memset(M, 0, sizeof(float)*n);
		memset(O, 0, sizeof(float)*n);
		gradientMagnitude(In, M, O, h, w, din, true);
		int binSize = 4; int nOrients = 9; int softBin = -1; float clip = 0.2f;
		int hb = h / binSize; int wb = w / binSize;
		int nb = hb*wb;
		int nChns = nOrients * 3 + 5;
		//float *H = new float[hb*wb*nChns];
		float *H = (float*)fftwf_malloc(sizeof(float)*hb*wb*nChns);
		memset(H, 0, sizeof(float)*hb*wb*nChns);
		fhog(M, O, H, h, w, binSize, nOrients, softBin, clip);
		// save hog features in row-major order
		int count = 0;
		for (int c = 0; c < dscale; c++)
			for (int i = 0; i < hb; i++)
				for (int j = 0; j < wb; j++) {
					scale_sample[count][s] = H[j + i*wb + nb*c] * scale_window[s];
					count++;
				}
		//delete[] H; delete[] M; delete[] O;
		fftwf_free(H); fftwf_free(M); fftwf_free(O);
	}

	void mosse_tracker::extract_image(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB,
		int pw, int ph, int xm, int xp, int ym, int yp, float *im_patch, int d)
	{
		// extract image with padding and save it in col-major order 
		if (d == 1) {
			if (ym >= 0 && yp <= imh - 1 &&
				xm >= 0 && xp <= imw - 1) {
				for (int i = ym; i < yp; i++)
					for (int j = xm; j < xp; j++)
					{
						im_patch[(i - ym) + (j - xm)*ph] = float(dataYorR[j + i*imw]);
					}
			}
			else {
				int xi, yi;
				for (int i = ym; i < yp; i++)
					for (int j = xm; j < xp; j++) {
						yi = i;  xi = j;
						if (j < 0) xi = 0;
						if (j > imw - 1) xi = imw - 1;
						if (i < 0) yi = 0;
						if (i > imh - 1) yi = imh - 1;
						im_patch[(i - ym) + (j - xm)*ph] = float(dataYorR[xi + yi*imw]);
					}
			}
		}
		else
			if (d == 3)
			{
				int c, r, n, n2; n = pw*ph; n2 = n*(d - 1);
				if (ym >= 0 && yp <= imh &&
					xm >= 0 && xp <= imw) {
					for (int i = ym; i < yp; i++)
						for (int j = xm; j < xp; j++)
						{
							r = (i - ym) + (j - xm)*ph; c = j + i*imw;
							im_patch[r] = float(dataYorR[c]);
							im_patch[r + n] = float(dataG[c]);
							im_patch[r + n2] = float(dataB[c]);
						}
				}
				else {
					int xi, yi;
					for (int i = ym; i < yp; i++)
						for (int j = xm; j < xp; j++)
						{
							yi = i;  xi = j;
							if (j < 0) xi = 0;
							if (j > imw - 1) xi = imw - 1;
							if (i < 0) yi = 0;
							if (i > imh - 1) yi = imh - 1;
							r = (i - ym) + (j - xm)*ph; c = xi + yi*imw;
							im_patch[r] = float(dataYorR[c]);
							im_patch[r + n] = float(dataG[c]);
							im_patch[r + n2] = float(dataB[c]);
						}
				}
			};
	};

	void mosse_tracker::get_translation_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, float **sample, int index) // data in row-major order
	{
		int patch_sz[2]; int w = sz[1]; int h = sz[0]; // size of the model with padding

		if (use_scale_4_translation_estimate == true) {
			patch_sz[0] = (int)floorf(float(h) * currentScaleFactor);
			patch_sz[1] = (int)floorf(float(w) * currentScaleFactor);
		}
		else {
			patch_sz[0] = h;
			patch_sz[1] = w;
		}

		// make sure the size is not to small
		if (patch_sz[0] <= 1) patch_sz[0] = 4; if (patch_sz[1] <= 1) patch_sz[1] = 4;

		int cx, cy;
		if (index != -1) {
			cx = ix + offset[index][1];
			cy = iy + offset[index][0];
		}
		else {
			cx = ix;
			cy = iy;
		}
		
		// find & check borders
		int xm = cx + 1 - (int)floor(float(patch_sz[1]) / 2); int xp = cx + 1 + patch_sz[1] - (int)floor(float(patch_sz[1]) / 2);
		int ym = cy + 1 - (int)floor(float(patch_sz[0]) / 2); int yp = cy + 1 + patch_sz[0] - (int)floor(float(patch_sz[0]) / 2);
		/*int xm = ix + 1 - (int)floor(float(patch_sz[1]) / 2); int xp = ix + 1 + patch_sz[1] - (int)floor(float(patch_sz[1]) / 2);
		int ym = iy + 1 - (int)floor(float(patch_sz[0]) / 2); int yp = iy + 1 + patch_sz[0] - (int)floor(float(patch_sz[0]) / 2);*/
		int pw = xp - xm; int ph = yp - ym;
		// extract image  with padding and save it in col-major order 
		int d = 0;
		if (dataYorR != 0 && dataG == 0 && dataB == 0) d = 1;
		if (dataYorR != 0 && dataG != 0 && dataB != 0) d = 3;
		float *im_patch = (float*)fftwf_malloc(sizeof(float) * pw * ph * d); // align is important	
		extract_image(dataYorR, dataG, dataB, pw, ph, xm, xp, ym, yp, im_patch, d);

		// resize image to model size (work with col-major order)
		float *resized_patch = (float*)fftwf_malloc(sizeof(float) * w * h * d);
		imResampleWrapper(im_patch, resized_patch, ph, pw, h, w, d, 1.0);

		//����� ��������� �������� ��������� ����� �����������
		//for (int i = 0; i < w * h * d; i++)
		//	//resized_patch[i] = roundf(resized_patch[i]);
		//	resized_patch[i] = float((int)(resized_patch[i] + 0.5f));

		// compute feature map	
		get_translation_feature_map(resized_patch, sample, h, w, d);

		fftwf_free(resized_patch);
		fftwf_free(im_patch);
	}

	void mosse_tracker::get_scale_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, float **scale_sample)
	{
		//float *resized_patch = (float*)fftwf_malloc(sizeof(float) * scale_model_sz[0] * scale_model_sz[1] * 3);  // never allocate memory here because tracker stops working properly
		for (int s = 0; s < nScales; s++)
		{
			int patch_sz[2]; int w = base_target_sz[1]; int h = base_target_sz[0]; // base size of the model
			patch_sz[0] = (int)floorf(h * (currentScaleFactor * scaleFactors[s]));
			patch_sz[1] = (int)floorf(w * (currentScaleFactor * scaleFactors[s]));

			//make sure the size is not to small
			if (patch_sz[0] <= 1) patch_sz[0] = 4; if (patch_sz[1] <= 1) patch_sz[1] = 4;

			//find & check borders
			int xm = ix + 1 - (int)floor(float(patch_sz[1]) / 2); int xp = ix + 1 + patch_sz[1] - (int)floor(float(patch_sz[1]) / 2);
			int ym = iy + 1 - (int)floor(float(patch_sz[0]) / 2); int yp = iy + 1 + patch_sz[0] - (int)floor(float(patch_sz[0]) / 2);

			int pw = xp - xm; int ph = yp - ym;
			int d = 0;
			if (dataYorR != 0 && dataG == 0 && dataB == 0) d = 1;
			if (dataYorR != 0 && dataG != 0 && dataB != 0) d = 3;
			float *im_patch = (float*)fftwf_malloc(sizeof(float) * pw * ph * d); // align is important	
			extract_image(dataYorR, dataG, dataB, pw, ph, xm, xp, ym, yp, im_patch, d);

			// �������� ������ ������ ��� ����� ��������� � ������ � ������ �������� �� ������� - ������� ��, ���-�� ������ �� imResampleWrapper
			float *resized_patch = (float*)fftwf_malloc(sizeof(float) * scale_model_sz[0] * scale_model_sz[1] * 3);
			// resize image to model size (work with col-major order)
			memset(resized_patch, 0, sizeof(float) * scale_model_sz[0] * scale_model_sz[1] * d);
			imResampleWrapper(im_patch, resized_patch, ph, pw, scale_model_sz[0], scale_model_sz[1], d, 1.0);

			//����� ��������� �������� ��������� ����� �����������
			for (int i = 0; i < scale_model_sz[0] * scale_model_sz[1] * d; i++)
				//resized_patch[i] = roundf(resized_patch[i]);
				resized_patch[i] = float((int)(resized_patch[i] + 0.5f));

			// compute feature map	
			get_scale_feature_map(resized_patch, scale_model_sz[0], scale_model_sz[1], d, scale_sample, s);

			fftwf_free(im_patch);
			fftwf_free(resized_patch);
		}
		//fftwf_free(resized_patch);
	}

	void mosse_tracker::extract_training_sample_info(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, bool first)
	{
		get_translation_sample(dataYorR, dataG, dataB, xt, -1);

		// calculate the translation filter update		
		for (int i = 0; i < dout; i++) {
			fftwf_execute(pxtf[i]);
		}

		// calculate numerator
		for (int i = 0; i < dout; i++)
			for (int j = 0; j < prodszhalf; j++) {
				num[i][j][0] = yf[j][0] * xtf[i][j][0] + yf[j][1] * xtf[i][j][1];
				num[i][j][1] = -yf[j][0] * xtf[i][j][1] + yf[j][1] * xtf[i][j][0];
			}

	    // kf = xtf.*conj(xtf) <-- idea 
		for (int i = 0; i < dout; i++) {	
			for (int j = 0; j < prodszhalf; j++) {
				kf[i][j] = xtf[i][j][0] * xtf[i][j][0] + xtf[i][j][1] * xtf[i][j][1] + lambda1;
			}
		}
			
		// obtain a sub-windows close to target for regression to 0
		for (int index = 0; index < 4; index++) // we use only 4 hard coded sub-windows 
		{
			extract_translation_test_sample(dataYorR, dataG, dataB, index); 
			// kfn(:,:,:,j) = conj(xfn) .*xfn; <-- idea
			for (int i = 0; i < dout; i++)
				for (int j = 0; j < prodszhalf; j++) {
					kf[i][j] += lambda2 * (xtf[i][j][0] * xtf[i][j][0] + xtf[i][j][1] * xtf[i][j][1]);
				}
		} 

		if (first == true) {
			for (int i = 0; i < dout; i++)
				for (int j = 0; j < prodszhalf; j++) {
					model_wf[i][j][0] = num[i][j][0] / kf[i][j];
					model_wf[i][j][1] = num[i][j][1] / kf[i][j];
				}
		}
		else {
			for (int i = 0; i < dout; i++)
				for (int j = 0; j < prodszhalf; j++) {
					model_wf[i][j][0] = comp_learning_rate * model_wf[i][j][0] + learning_rate * (num[i][j][0] / kf[i][j]);
					model_wf[i][j][1] = comp_learning_rate * model_wf[i][j][1] + learning_rate * (num[i][j][1] / kf[i][j]);
				}
		}

		// calculate the scale filter update	
		get_scale_sample(dataYorR, dataG, dataB, xs);

		for (int i = 0; i < sizess; i++) {
			fftwf_execute(pxsf[i]);
		}

		if (first == true) {

			// calculate numerator
			for (int i = 0; i < sizess; i++)
				for (int j = 0; j < (nScales / 2 + 1); j++) {
					//new_sf_num = ysf .* conj(xsf) <-- idea 
					sf_num[i][j][0] = ysf[j][0] * xsf[i][j][0] + ysf[j][1] * xsf[i][j][1];
					sf_num[i][j][1] = -ysf[j][0] * xsf[i][j][1] + ysf[j][1] * xsf[i][j][0];
				}

			// calculate denominator
			memset(sf_den, 0, sizeof(float) * (nScales / 2 + 1));
			for (int i = 0; i < sizess; i++)
				for (int j = 0; j < (nScales / 2 + 1); j++) {
					//new_hf_den = sum(xsf .* conj(xsf), 1) <-- idea 
					sf_den[j] += xsf[i][j][0] * xsf[i][j][0] + xsf[i][j][1] * xsf[i][j][1];
				}
		}

		else {

			// calculate numerator
			for (int i = 0; i < sizess; i++)
				for (int j = 0; j < (nScales / 2 + 1); j++) {
					//new_sf_num = ysf .* conj(xsf) <-- idea 
					sf_num[i][j][0] = (1 - learning_rate) * sf_num[i][j][0] + learning_rate * (ysf[j][0] * xsf[i][j][0] + ysf[j][1] * xsf[i][j][1]);
					sf_num[i][j][1] = (1 - learning_rate) * sf_num[i][j][1] + learning_rate * (-ysf[j][0] * xsf[i][j][1] + ysf[j][1] * xsf[i][j][0]);
				}

			// calculate denominator
			float sum_sd = 0;
			for (int j = 0; j < (nScales / 2 + 1); j++) {
				sum_sd = 0;
				for (int i = 0; i < sizess; i++) {
					//new_hf_den = sum(xsf .* conj(xsf), 1) <-- idea 
					sum_sd += xsf[i][j][0] * xsf[i][j][0] + xsf[i][j][1] * xsf[i][j][1];
				}
				sf_den[j] = (1 - learning_rate) * sf_den[j] + learning_rate * sum_sd;
			}
		}
	}

	void mosse_tracker::extract_translation_test_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, int index)
	{
		// extract the test sample feature map for the translation filter
		get_translation_sample(dataYorR, dataG, dataB, xt, index);

		// calculate the correlation response of the translation filter			
		for (int i = 0; i < dout; i++) {
			fftwf_execute(pxtf[i]);
		}
	}

	void mosse_tracker::extract_scale_test_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB)
	{
		// extract the test sample feature map for the scale filter
		get_scale_sample(dataYorR, dataG, dataB, xs);

		// calculate the correlation response of the scale filter			
		for (int i = 0; i < sizess; i++) {
			fftwf_execute(pxsf[i]);
		}
	}

	bool mosse_tracker::findNextLocation(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB)
	{
		//  ***************** TRANSLATION PART *************************//

		// extract test sample
		extract_translation_test_sample(dataYorR, dataG, dataB, -1);

		//find responce map
		memset(rt, 0, sizeof(fftwf_complex) * prodszhalf);
		
		//sum(model_wf.*xtf, 3)
		for (int i = 0; i < dout; i++)
			for (int j = 0; j < prodszhalf; j++) {
				rt[j][0] += model_wf[i][j][0] * xtf[i][j][0] - model_wf[i][j][1] * xtf[i][j][1];
				rt[j][1] += model_wf[i][j][0] * xtf[i][j][1] + model_wf[i][j][1] * xtf[i][j][0];
			}

		fftwf_execute(prespt);
		
		//for (int j = 0; j <prodsz; j++) respt[j] /= (csz[0] * csz[1]); // you can safely uncomment this line

		// find the maximum translation response
		int mr = 0; int mc = 0; score = respt[0];
		for (int i = 0; i < csz[0]; i++)
			for (int j = 0; j < csz[1]; j++)
			{
				float cv = respt[j + i*csz[1]];
				if (cv > score) { score = cv; mr = i; mc = j; }
			}

		score /= (csz[0] * csz[1]);

		//if (isnan(score)) //stop!

		if (mr > (csz[0] / 2)) //wrap around to negative half - space of vertical axis
			mr -= csz[0];
		
		if (mc > (csz[1] / 2)) //wrap around to negative half - space of horizontal axis
			mc -= csz[1];

		// update the position
		if (use_scale_4_translation_estimate == true) {
			ix += (int)roundf(cell_size * mc * currentScaleFactor);
			iy += (int)roundf(cell_size * mr * currentScaleFactor);
		}
		else {
			ix += cell_size * mc;
			iy += cell_size * mr;
		}

		//  ***************** SCALE PART *************************//

		// ������ ������� ������������ ������ ��������� ix, iy
		extract_scale_test_sample(dataYorR, dataG, dataB);

		// find the maximum scale response
		memset(rts, 0, sizeof(fftwf_complex)*(nScales / 2 + 1));
		for (int i = 0; i < sizess; i++)
			for (int j = 0; j < (nScales / 2 + 1); j++) {
				//sum(sf_num .* xsf, 1)
				rts[j][0] += sf_num[i][j][0] * xsf[i][j][0] - sf_num[i][j][1] * xsf[i][j][1];
				rts[j][1] += sf_num[i][j][0] * xsf[i][j][1] + sf_num[i][j][1] * xsf[i][j][0];
			}

		for (int j = 0; j < (nScales / 2 + 1); j++) {
			//real(ifft(sum(...) ./ (sf_den + lambda)));
			rts[j][0] /= (sf_den[j] + lambda1);
			rts[j][1] /= (sf_den[j] + lambda1);
		}

		fftwf_execute(presps);

		//for (int j = 0; j < nScales; j++) resps[j] /= nScales; // you can safely uncomment this line

		// find the maximum scale response
		int indx = 0; double max_val = resps[0];
		for (int i = 0; i < nScales; i++)
		{
			float cv = resps[i];
			if (resps[i] > max_val) { max_val = resps[i]; indx = i; }
		}

		currentScaleFactor *= scaleFactors[indx];

		if (currentScaleFactor < min_scale_factor)
			currentScaleFactor = min_scale_factor;
		else
			if (currentScaleFactor > max_scale_factor)
				currentScaleFactor = max_scale_factor;

		//reset offsets according to currentScaleFactor
		if (use_scale_4_translation_estimate == true) {
			offset[0][0] = (int)floorf(float(-base_target_sz[0]) * currentScaleFactor);
			offset[1][1] = (int)floorf(float(-base_target_sz[1]) * currentScaleFactor);
			offset[2][0] = (int)floorf(float(base_target_sz[0]) * currentScaleFactor);
			offset[3][1] = (int)floorf(float(base_target_sz[1]) * currentScaleFactor);
		}

		// extract training sample
		extract_training_sample_info(dataYorR, dataG, dataB, false);

		// calculate the new target size	
		target_size[0] = (int)floorf((float(base_target_sz[0]) * currentScaleFactor));
		target_size[1] = (int)floorf((float(base_target_sz[1]) * currentScaleFactor));

		// update the size
		if (use_scale_4_translation_estimate == true) {
			target_size[0] = (int)floorf((float(base_target_sz[0]) * currentScaleFactor));
			target_size[1] = (int)floorf((float(base_target_sz[1]) * currentScaleFactor));
		}
		else {
			target_size[0] = base_target_sz[0];
			target_size[1] = base_target_sz[1];
		}

		return true;
	}

	bool mosse_tracker::getNewLocationCoordinates(int &x, int &y, int &w, int &h, float &scr)
	{
		if (true) {
			x = ix;
			y = iy;
			w = target_size[1];
			h = target_size[0];
			scr = score;
			return true;
		}
		else
			return false;
	}
};



