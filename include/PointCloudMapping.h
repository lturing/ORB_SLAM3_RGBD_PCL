/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "KeyFrame.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>

using namespace ORB_SLAM3;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PointCloudData
{
public:
    PointCloudData();
    ~PointCloudData();
public:
    PointCloud::Ptr cloud;
public:
    KeyFrame* kf;
      
};


class PointCloudMapping
{
public:

    PointCloudMapping( double resolution_ );

    // 插入一个keyframe，会更新一次地图
    //void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, const std::vector<KeyFrame*>& KFs, cv::Mat& mK, cv::Mat& mDistCoef);

    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, const std::vector<KeyFrame*>& KFs);
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);

    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, const std::vector<KeyFrame*>& KFs, cv::Mat& mK, cv::Mat& mDistCoef);

    void shutdown();
    void viewer();

protected:

    PointCloud::Ptr globalMap;
    std::thread* viewerThread;
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;

    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    mutex                   keyframeMutex;

    double resolution = 0.04;
    pcl::VoxelGrid<PointT>  voxel;

    std::vector<PointCloudData> mPointCloudDatas;

    std::vector<KeyFrame*> mvpKFs;

    vector<KeyFrame*>   keyframes;
    vector<cv::Mat>     colorImgs;
    vector<cv::Mat>     depthImgs;

};

#endif // POINTCLOUDMAPPING_H