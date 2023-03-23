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

#include "PointCloudMapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"

#include <boost/make_shared.hpp>

PointCloudData::PointCloudData(){
    cloud = std::make_shared<PointCloud>();
}

PointCloudData::~PointCloudData()
{
}


PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = std::make_shared<pcl::PointCloud<PointT>>();

    viewerThread = new thread(&PointCloudMapping::viewer, this );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

/*
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, const std::vector<KeyFrame*>& KFs)
{
    //cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );

    mvpKFs = KFs;

    keyFrameUpdated.notify_one();
}


PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }
    tmp->width = tmp->points.size();
    tmp->height = 1; 
    tmp->is_dense = false;

    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());

    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        std::set<long unsigned int> sKFs;
        for (auto kf : mvpKFs)
            sKFs.insert(kf->mnFrameId);

        for ( size_t i=0; i<N ; i++ )
        {
            if (keyframes[i]->isBad())
                    continue;
                                
            if (sKFs.find(keyframes[i]->mnFrameId) == sKFs.end())
                    continue; 

            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
            *globalMap += *p;
        }
        
        //PointCloud::Ptr tmp(new PointCloud());
        //voxel.setInputCloud( globalMap );
        //voxel.filter( *tmp );
        //globalMap->swap( *tmp );
        
        viewer.showCloud( globalMap );
        //cout << "show global map, size=" << globalMap->points.size() << endl;
    }
}

*/


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, const std::vector<KeyFrame*>& KFs, cv::Mat& mK, cv::Mat& mDistCoef)
{
    //cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);

    std::vector<cv::KeyPoint> mvKeys;
    int stride = 6;
    for (int i = 0; i < color.rows; i += stride)
    {
        for (int j = 0; j < color.cols; j+= stride)
        {
            cv::KeyPoint kp;
            kp.pt.x = j;
            kp.pt.y = i;
            mvKeys.push_back(kp);
        }
    }

    int N = mvKeys.size();

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat=mat.reshape(1);

    int cnt = 0;
    PointCloud::Ptr cloud( new PointCloud() );
    float v_min = 1000000000.0;
    float v_max = 0.0;

    for ( int m=0; m<depth.rows; m+=stride )
    {
        for ( int n=0; n<depth.cols; n+=stride )
        {
            float d = depth.ptr<float>(m)[n];
            v_min = v_min > d ? d : v_min; 
            v_max = v_max > d ? v_max : isnan(d) ? v_max : d;

            if (isnan(d) || d < 0.01 || d>10)
            {
                cnt += 1;
                continue;
            }
            
            PointT p;
            p.x = ( mat.at<float>(cnt, 0) - kf->cx) * d / kf->fx;
            p.y = ( mat.at<float>(cnt, 1) - kf->cy) * d / kf->fy;
            p.z = d;
            //p.b = color.ptr<uchar>(m)[n*3];
            //p.g = color.ptr<uchar>(m)[n*3+1];
            //p.r = color.ptr<uchar>(m)[n*3+2];
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back(p);
            cnt += 1;
        }
    }


    cloud->width = cloud->points.size();
    cloud->height = 1;

    PointCloudData pcd;
    pcd.kf = kf;
    pcd.cloud = cloud;

    mPointCloudDatas.push_back(pcd);

    mvpKFs = KFs;

    keyFrameUpdated.notify_one();
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            
            std::set<long unsigned int> sKFs;
            for (auto kf : mvpKFs)
                sKFs.insert(kf->mnFrameId);
            
            PointCloud::Ptr globalMap = std::make_shared<pcl::PointCloud<PointT>>();
            for ( auto pcd : mPointCloudDatas)
            {
                if (pcd.kf->isBad())
                    continue;
                                
                if (sKFs.find(pcd.kf->mnFrameId) == sKFs.end())
                    continue; 


                Eigen::Isometry3d T = Converter::toSE3Quat( pcd.kf->GetPoseInverse() );
                PointCloud::Ptr cloud(new PointCloud);
                pcl::transformPointCloud(*pcd.cloud, *cloud, T.matrix()); 

                *globalMap += *cloud;
            }
            //globalMap->width = globalMap->points.size();
            //globalMap->height = 1;
            //globalMap->is_dense = true;
            
            //PointCloud::Ptr tmp(new PointCloud());
            //voxel.setInputCloud( globalMap );
            //voxel.filter( *tmp );
            //globalMap->swap( *tmp );
            
            viewer.showCloud( globalMap );
            //cout << "show global map, size=" << globalMap->points.size() << endl;
        }
    }
}

