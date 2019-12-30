#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkAutoInit.h>

struct XYZ {
	float
		x_min = INT_MAX, x_max = INT_MIN,
		y_min = INT_MAX, y_max = INT_MIN,
		z_min = INT_MAX, z_max = INT_MIN;
};
struct COUNT {
	float
		x_count = 0,
		y_count = 0,
		z_count = 0;
};
VTK_MODULE_INIT(vtkRenderingOpenGL);
using namespace std;

void show_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 255, 0, 0);//关键点
	viewer->addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normal, 2, 5, "normals");
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
void show_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal,
	vector<vector<vector<vector<int>>>>& voxel, vector<vector<vector<bool>>>& flag) {

	pcl::PointCloud<pcl::Normal>::Ptr normal_temp(new pcl::PointCloud<pcl::Normal>);
	*normal_temp = *normal;
	for (int i = 0; i < voxel.size(); i++) {
		for (int j = 0; j < voxel[0].size(); j++) {
			for (int k = 0; k < voxel[0][0].size(); k++) {
				if (!flag[i][j][k]) {
					for (int m = 0; m < voxel[i][j][k].size(); m++) {
						normal_temp->points[voxel[i][j][k][m]].normal_x = 0;
						normal_temp->points[voxel[i][j][k][m]].normal_y = 0;
						normal_temp->points[voxel[i][j][k][m]].normal_z = 0;
					}
				}

			}
		}
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 255, 0, 0);//关键点
	viewer->addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normal_temp, 2, 5, "normals");
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void filte_r(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float r) {
	pcl::RadiusOutlierRemoval<pcl::PointXYZ>  rout;
	rout.setInputCloud(cloud);
	rout.setRadiusSearch(3.0f*r);//设置搜索半径的值
	rout.setMinNeighborsInRadius(10);//设置最小邻居个数，默认是1
	rout.filter(*cloud);
}

void normal_to_one(pcl::Normal& normal) {
	float res = sqrt(pow(normal.normal_x, 2) + pow(normal.normal_y, 2) + pow(normal.normal_z, 2));
	normal.normal_x = normal.normal_x / res;
	normal.normal_y = normal.normal_y / res;
	normal.normal_z = normal.normal_z / res;
}

pcl::PointCloud<pcl::Normal>::Ptr normal_est(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float r) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(r);
	//ne.setKSearch(k);
	ne.compute(*normals);
	return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr com_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float r) {

	pcl::PointCloud<pcl::Normal>::Ptr n1(new pcl::PointCloud<pcl::Normal>());
	*n1 = *normal_est(cloud, 3.0*r);

	pcl::PointCloud<pcl::Normal>::Ptr n2(new pcl::PointCloud<pcl::Normal>());
	*n2 = *normal_est(cloud, 4.0*r);

	pcl::PointCloud<pcl::Normal>::Ptr n3(new pcl::PointCloud<pcl::Normal>());
	*n3 = *normal_est(cloud, 5.0*r);

	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	for (int i = 0; i < cloud->size(); i++) {
		pcl::Normal n;
		n.normal_x = 0.5*n1->points[i].normal_x + n2->points[i].normal_x + 0.5*n3->points[i].normal_x;
		n.normal_y = 0.5*n1->points[i].normal_y + n2->points[i].normal_y + 0.5*n3->points[i].normal_y;
		n.normal_z = 0.5*n1->points[i].normal_z + n2->points[i].normal_z + 0.5*n3->points[i].normal_z;
		normal->push_back(n);
	}
	for (int i = 0; i < normal->size(); i++) {
		normal_to_one(normal->points[i]);
	}
	return normal;
}

void com_filter_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal) {

	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	tree->setInputCloud(cloud);

	for (int i = 0; i < cloud->size(); i++) {
		std::vector<int> id;
		std::vector<float> dis;
		tree->nearestKSearch(cloud->points[i], 10, id, dis);
		for (int j = 0; j < id.size(); j++) {
			normal->points[i].normal_x += normal->points[id[j]].normal_x;
			normal->points[i].normal_y += normal->points[id[j]].normal_y;
			normal->points[i].normal_z += normal->points[id[j]].normal_z;
		}
		normal_to_one(normal->points[i]);
	}
	return;
}


XYZ com_box(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float r) {
	XYZ xyz;
	for (int i = 0; i < cloud->size(); i++) {
		xyz.x_min = min(xyz.x_min, cloud->points[i].x);
		xyz.x_max = max(xyz.x_max, cloud->points[i].x);

		xyz.y_min = min(xyz.y_min, cloud->points[i].y);
		xyz.y_max = max(xyz.y_max, cloud->points[i].y);

		xyz.z_min = min(xyz.z_min, cloud->points[i].z);
		xyz.z_max = max(xyz.z_max, cloud->points[i].z);
	}
	return xyz;
}
COUNT com_count(XYZ xyz, float r) {
	COUNT count;
	count.x_count = ceil((xyz.x_max - xyz.x_min) / r);
	count.y_count = ceil((xyz.y_max - xyz.y_min) / r);
	count.z_count = ceil((xyz.z_max - xyz.z_min) / r);
	return count;
}
pcl::PointXYZ com_center(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::PointXYZ center;
	if (cloud->size() == 0)
		return center;
	for (int i = 0; i < cloud->size(); i++) {
		center.x += cloud->points[i].x;
		center.y += cloud->points[i].y;
		center.z += cloud->points[i].z;
	}
	center.x /= cloud->size();
	center.y /= cloud->size();
	center.z /= cloud->size();
	return center;
}
pcl::PointXYZ com_center(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<int>& id) {
	pcl::PointXYZ center;
	if (id.size() == 0)
		return center;
	for (int i = 0; i < id.size(); i++) {
		center.x += cloud->points[id[i]].x;
		center.y += cloud->points[id[i]].y;
		center.z += cloud->points[id[i]].z;
	}
	center.x /= id.size();
	center.y /= id.size();
	center.z /= id.size();
	return center;
}
float com_dis(pcl::PointXYZ p1, pcl::PointXYZ p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}
float com_dis(pcl::PointXYZ p, pcl::PointXYZ center, pcl::PointXYZ pice_center) {
	float a, b, c;
	a = com_dis(center, pice_center);
	b = com_dis(p, center);
	c = com_dis(p, pice_center);
	float s = 1.0f / 4.0f*sqrt((a + b + c)*(a + b - c)*(a + c - b)*(b + c - a));
	return 2.0f*s / a;
}
void reverse_normal(pcl::Normal& normal) {
	normal.normal_x = -normal.normal_x;
	normal.normal_y = -normal.normal_y;
	normal.normal_z = -normal.normal_z;
}
float pot(pcl::Normal& view_vec, pcl::Normal& normal) {
	return view_vec.normal_x*normal.normal_x +
		view_vec.normal_y*normal.normal_y +
		view_vec.normal_z*normal.normal_z;
}
pcl::Normal com_view_vec(pcl::PointXYZ start, pcl::PointXYZ end) {
	pcl::Normal view_vec;
	view_vec.normal_x = end.x - start.x;
	view_vec.normal_y = end.y - start.y;
	view_vec.normal_z = end.z - start.z;
	return view_vec;
}
pcl::Normal com_avg_normal(pcl::PointCloud<pcl::Normal>::Ptr normal, vector<int>& id) {
	pcl::Normal avg_normal;
	if (id.size() == 0) {
		avg_normal.normal_x = 0;
		avg_normal.normal_y = 0;
		avg_normal.normal_z = 0;
		return avg_normal;
	}
	for (int i = 0; i < id.size(); i++) {
		avg_normal.normal_x += normal->points[id[i]].normal_x;
		avg_normal.normal_y += normal->points[id[i]].normal_y;
		avg_normal.normal_z += normal->points[id[i]].normal_z;
	}
	normal_to_one(avg_normal);
	return avg_normal;
}

void com_in_re_direct(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, vector<int>& id,
	pcl::Normal& avg_normal, pcl::PointXYZ center) {
	for (int i = 0; i < id.size(); i++) {
		pcl::Normal view_vec = com_view_vec(center, cloud->points[id[i]]);
		if (pot(view_vec, normal->points[id[i]]) < 0) {
			reverse_normal(normal->points[id[i]]);
			reverse_normal(avg_normal);
		}
	}
	return;
}


void com_out_re_direct(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, vector<int>& id,
	pcl::Normal& avg_normal, pcl::PointXYZ center) {
	for (int i = 0; i < id.size(); i++) {
		pcl::Normal view_vec = com_view_vec(center, cloud->points[id[i]]);
		if (pot(view_vec, normal->points[id[i]]) > 0) {
			reverse_normal(normal->points[id[i]]);
			reverse_normal(avg_normal);
		}
	}
	return;
}

bool judge(vector<vector<vector<vector<int>>>>& voxel, int i, int j, int k) {
	return (i >= 0 && i < voxel.size() && j >= 0 && j < voxel[0].size() && k >= 0 && k < voxel[0][0].size());
}

bool com_know_re_direct(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, vector<int>& id, pcl::Normal& avg_nromal, pcl::PointXYZ center) {

	if (id.size() == 0)
		return true;
	pcl::PointXYZ pice_center = com_center(cloud, id);
	vector<int> J;
	for (int i = 0; i < id.size(); i++) {
		if (com_dis(cloud->points[id[i]], center, pice_center) <= 1.5f) {
			J.push_back(id[i]);
		}
	}
	int in = 0, out = 0;//////////////////in1,out2,unkonw0////////////////
	float dis_c_c = com_dis(center, pice_center);
	for (int i = 0; i < J.size(); i++) {
		float dis_p_c = com_dis(center, cloud->points[J[i]]);
		if (dis_c_c <= dis_p_c)
			in = 1;
		if (dis_c_c >= dis_p_c)
			out = 1;
	}
	if (in == out || id.size() < 50) {
		return false;
	}
	else {
		if (in == 1) {
			com_in_re_direct(cloud, normal, id, avg_nromal, pice_center);
		}
		else if (out == 1) {
			com_out_re_direct(cloud, normal, id, avg_nromal, pice_center);
		}
		return true;
	}
}

bool com_next_re_direct(pcl::PointCloud<pcl::Normal>::Ptr normal,
	vector<int>& id, vector<int>& next_id,
	pcl::Normal& avg_nromal, pcl::Normal& next_avg_normal) {
	if (pot(avg_nromal, next_avg_normal) < 0) {
		for (int i = 0; i < id.size(); i++) {
			reverse_normal(normal->points[id[i]]);
		}
		reverse_normal(avg_nromal);
	}

	return true;
}

bool com_unkonw_re_direct(pcl::PointCloud<pcl::Normal>::Ptr normal,
	vector<vector<vector<vector<int>>>>& voxel,
	vector<vector<vector<bool>>>& flag,
	vector<vector<vector<pcl::Normal>>>& avg_normal,
	int i, int j, int k) {

	vector<vector<int>> vec = {
		{1,0,0},{-1,0,0},
		{0,1,0},{0,-1,0},
		{0,0,1},{0,0,-1},
		//{1, 1, 0}, { -1,1,0 }, { 1,-1,0 }, { -1,-1,0 },
		//{ 1,0,1 }, { -1,0,1 }, { 1,0,-1 }, { -1,0,-1 },
		//{ 0,1,1 }, { 0,-1,1 }, { 0,1,-1 }, { 0,-1,-1 },
		//{1, 1, 1}, { -1,1,1 }, { 1,-1,1 }, { 1,1,-1 }, { -1,-1,1 }, { -1,1,-1 }, { 1,-1,-1 }, { -1,-1,-1 }
	};

	int zero_count = 0, count = 0;
	for (int m = 0; m < vec.size(); m++) {
		if (judge(voxel, i + vec[m][0], j + vec[m][1], k + vec[m][2])) {
			count++;
			if (voxel[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]].size() == 0) {
				zero_count++;
				continue;
			}
			if (flag[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]] &&
				com_next_re_direct(normal, voxel[i][j][k], voxel[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]],
				avg_normal[i][j][k], avg_normal[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]])) {
				return true;
			}
		}
	}
	if (zero_count == count)
		return true;
	else
		return false;
}

void com_normal_re_direct(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, float r) {

	XYZ xyz = com_box(cloud, r);
	COUNT count = com_count(xyz, r);
	pcl::PointXYZ center = com_center(cloud);

	vector<vector<vector<vector<int>>>> voxel(count.x_count, vector<vector<vector<int>>>
		(count.y_count, vector<vector<int>>(count.z_count)));

	vector<vector<vector<bool>>> flag(count.x_count, vector<vector<bool>>
		(count.y_count, vector<bool>(count.z_count, false)));

	vector<vector<vector<pcl::Normal>>> avg_normal(count.x_count, vector<vector<pcl::Normal>>
		(count.y_count, vector<pcl::Normal>(count.z_count)));

	int c = 0, total = count.x_count*count.y_count*count.z_count;
	for (int i = 0; i < cloud->size(); i++) {
		int idx, idy, idz;
		idx = floor((cloud->points[i].x - xyz.x_min) / r);
		idy = floor((cloud->points[i].y - xyz.y_min) / r);
		idz = floor((cloud->points[i].z - xyz.z_min) / r);
		voxel[idx][idy][idz].push_back(i);
	}
	for (int i = 0; i < count.x_count; i++) {
		for (int j = 0; j < count.y_count; j++) {
			for (int k = 0; k < count.z_count; k++) {
				if (voxel[i][j][k].size() != 0) {
					avg_normal[i][j][k] = com_avg_normal(normal, voxel[i][j][k]);
					for (int m = 0; m < voxel[i][j][k].size(); m++) {
						if (pot(normal->points[voxel[i][j][k][m]], avg_normal[i][j][k]) < 0) {
							reverse_normal(normal->points[voxel[i][j][k][m]]);
						}
					}
					avg_normal[i][j][k] = com_avg_normal(normal, voxel[i][j][k]);
				}
			}
		}
	}

	for (int i = 0; i < count.x_count; i++) {
		for (int j = 0; j < count.y_count; j++) {
			for (int k = 0; k < count.z_count; k++) {
				if (com_know_re_direct(cloud, normal, voxel[i][j][k], avg_normal[i][j][k], center)) {
					flag[i][j][k] = true;
					c++;
				}
			}
		}
	}
	//show_normal(cloud, normal, voxel, flag);
	for (int i = 0; i < count.x_count; i++) {
		for (int j = 0; j < count.y_count; j++) {
			for (int k = 0; k < count.z_count; k++) {
				vector<vector<int>> vec = {
						{1,0,0},{-1,0,0},
						{0,1,0},{0,-1,0},
						{0,0,1},{0,0,-1},
						//{1, 1, 0}, { -1,1,0 }, { 1,-1,0 }, { -1,-1,0 },
						//{1,0,1 }, { -1,0,1 }, { 1,0,-1 }, { -1,0,-1 },
						//{0,1,1 }, { 0,-1,1 }, { 0,1,-1 }, { 0,-1,-1 },
						//{1, 1, 1}, { -1,1,1 }, { 1,-1,1 }, { 1,1,-1 }, { -1,-1,1 }, { -1,1,-1 }, { 1,-1,-1 }, { -1,-1,-1 }
				};
				for (int m = 0; m < vec.size(); m++) {
					if (flag[i][j][k] &&
						judge(voxel, i + vec[m][0], j + vec[m][1], k + vec[m][2]) &&
						pot(avg_normal[i][j][k], avg_normal[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]]) < 0) {
						flag[i][j][k] = false;
						c--;
						break;
					}
				}
			}
		}
	}
	//show_normal(cloud, normal, voxel, flag);
	for (int i = 0; i < count.x_count; i++) {
		for (int j = 0; j < count.y_count; j++) {
			for (int k = 0; k < count.z_count; k++) {
				if (voxel[i][j][k].size() != 0) {
					avg_normal[i][j][k] = com_avg_normal(normal, voxel[i][j][k]);
					for (int m = 0; m < voxel[i][j][k].size(); m++) {
						if (pot(normal->points[voxel[i][j][k][m]], avg_normal[i][j][k]) < 0) {
							reverse_normal(normal->points[voxel[i][j][k][m]]);
						}
					}
					avg_normal[i][j][k] = com_avg_normal(normal, voxel[i][j][k]);
				}
			}
		}
	}
	//show_normal(cloud, normal, voxel, flag);
	while (c < total) {
		for (int i = 0; i < count.x_count; i++) {
			for (int j = 0; j < count.y_count; j++) {
				for (int k = 0; k < count.z_count; k++) {
					//未知方向且可根据周围体素定向，置为已知
					if (!flag[i][j][k] &&
						com_unkonw_re_direct(normal, voxel, flag, avg_normal, i, j, k)) {
						flag[i][j][k] = true;
						c++;
					}
				}
			}
		}
		//show_normal(cloud, normal, voxel, flag);
	}
	//for (int i = 0; i < count.x_count; i++) {
	//	for (int j = 0; j < count.y_count; j++) {
	//		for (int k = 0; k < count.z_count; k++) {
	//			vector<vector<int>> vec = {
	//					{1,0,0},{-1,0,0},
	//					{0,1,0},{0,-1,0},
	//					{0,0,1},{0,0,-1},
	//					{1, 1, 0}, { -1,1,0 }, { 1,-1,0 }, { -1,-1,0 },
	//					{ 1,0,1 }, { -1,0,1 }, { 1,0,-1 }, { -1,0,-1 },
	//					{ 0,1,1 }, { 0,-1,1 }, { 0,1,-1 }, { 0,-1,-1 },
	//					{1, 1, 1}, { -1,1,1 }, { 1,-1,1 }, { 1,1,-1 }, { -1,-1,1 }, { -1,1,-1 }, { 1,-1,-1 }, { -1,-1,-1 }
	//			};
	//			for (int m = 0; m < vec.size(); m++) {
	//				if (flag[i][j][k] &&
	//					judge(voxel, i + vec[m][0], j + vec[m][1], k + vec[m][2]) &&
	//					pot(avg_normal[i][j][k], avg_normal[i + vec[m][0]][j + vec[m][1]][k + vec[m][2]]) < 0) {
	//					flag[i][j][k] = false;
	//					c--;
	//					break;
	//				}
	//			}
	//		}
	//	}
	//}
	//show_normal(cloud, normal, voxel, flag);
	//while (c < total) {
	//	for (int i = 0; i < count.x_count; i++) {
	//		for (int j = 0; j < count.y_count; j++) {
	//			for (int k = 0; k < count.z_count; k++) {
	//				//未知方向且可根据周围体素定向，置为已知
	//				if (!flag[i][j][k] &&
	//					com_unkonw_re_direct(normal, voxel, flag, avg_normal, i, j, k)) {
	//					flag[i][j][k] = true;
	//					c++;
	//				}
	//			}
	//		}
	//	}
	//	show_normal(cloud, normal, voxel, flag);
	//}
	//for (int i = 0; i < count.x_count; i++) {
	//	for (int j = 0; j < count.y_count; j++) {
	//		for (int k = 0; k < count.z_count; k++) {
	//			if (!flag[i][j][k]) {
	//				for (int m = 0; m < voxel[i][j][k].size(); m++) {
	//					normal->points[voxel[i][j][k][m]].normal_x = 0;
	//					normal->points[voxel[i][j][k][m]].normal_y = 0;
	//					normal->points[voxel[i][j][k][m]].normal_z = 0;
	//				}
	//			}
	//		}
	//	}
	//}

	return;
}

int main() {
	string cloud_name;
	float leaf_size = 1.0f;
	float voxel = 0;
	while (cin >> cloud_name >> voxel) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/filter/" + cloud_name + ".ply", *cloud);
		filte_r(cloud, leaf_size);
		pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
		double start, end;
		start = GetTickCount();
		*normal = *normal_est(cloud, 5.0*leaf_size);
		//*normal = *com_normal(cloud, leaf_size);
		//com_filter_normal(cloud, normal);
		show_normal(cloud, normal);
		com_normal_re_direct(cloud, normal, voxel*leaf_size);
		end = GetTickCount();
		cout << "法线计算耗时：" << end - start << "ms" << endl;
		show_normal(cloud, normal);
	}
	return 0;
}