#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "opencv2/opencv.hpp"
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlfcn.h>

#include"utils.h"

#include <tuple>

using namespace cv;
using namespace std;

bool save_tensor_txt(torch::Tensor tensor_in_,string path_txt)
{
#include "fstream"
    ofstream outfile(path_txt);
    torch::Tensor tensor_in = tensor_in_.clone();
    tensor_in = tensor_in.view({-1,1});
    tensor_in = tensor_in.to(torch::kCPU);

    auto result_data = tensor_in.accessor<float, 2>();

    for(int i=0;i<result_data.size(0);i++)
    {
        float val = result_data[i][0];
        //        std::cout<<"val="<<val<<std::endl;
        outfile<<val<<std::endl;

    }

    return true;
}

torch::Tensor get_adj_ind(int n_adj, int n_nodes, torch::Device device)
{
    //torch::Device m_device(torch::kCUDA);
    vector<int>v_i;
    for(int i = -n_adj/2;i<n_adj/2+1;i++)
    {
        if(0 == i) {continue;}
        v_i.push_back(i);
    }
    torch::Tensor ind = torch::tensor(v_i).unsqueeze(0).to(torch::kInt8);
    //    ind.print();
    torch::Tensor aa = torch::arange(n_nodes).unsqueeze(1).to(torch::kInt8);
    //    aa.print();

    torch::Tensor merge = aa + ind;
    merge = merge.to(torch::kLong);
    //    std::cout<<merge<<std::endl;
    //    std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    merge = merge % n_nodes;
    //    std::cout<<merge<<std::endl;
    return merge.to(torch::kLong).to(device);
}


torch::Tensor uniform_upsample(torch::Tensor &poly, int p_num)
{
    torch::Tensor next_poly = torch::roll(poly, -1, 2);
    torch::Tensor edge_len = (next_poly - poly).pow(2).sum(3).sqrt();
    //    next_poly.print();
    //    std::cout<<next_poly[0][0][0][1]<<std::endl;
    //    edge_len.print();
    torch::Tensor a = torch::sum(edge_len, 2);
    a.unsqueeze_(2);
    //    a.print();

    //    edge_len.print();
    torch::Tensor b = edge_len * p_num / a;
    torch::Tensor edge_num = torch::round(b).to(torch::kLong);
    //    std::cout<<edge_num[0][17][0]<<std::endl;
    //    std::cout<<edge_num[0][22][1]<<std::endl;
    //    std::cout<<edge_num[0][44][2]<<std::endl;
    //    std::cout<<edge_num[0][2][3]<<std::endl;
    //    edge_num.print();

    edge_num.clamp_(1,INT_MAX*1.0);
    torch::Tensor edge_num_sum = torch::sum(edge_num, 2);

//    torch::Tensor edge_idx_sort2 = torch::argsort(edge_num, 2, true); //################################## yhl
//    torch::Tensor edge_idx_sort;

    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(edge_num, 2, true);
//    torch::Tensor v = std::get<0>(sort_ret);
    torch::Tensor edge_idx_sort = std::get<1>(sort_ret);



    calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num);


    edge_num_sum = torch::sum(edge_num, 2);
    torch::Tensor edge_start_idx = torch::cumsum(edge_num, 2) - edge_num;
    //    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    auto tmp_1 = calculate_wnp(edge_num, edge_start_idx, p_num);

    torch::Tensor weight = std::get<0>(tmp_1);
    //    topk_scores.print();
    torch::Tensor ind = std::get<1>(tmp_1);
    //    weight.print();
    //    ind.print();
    //
    //    std::cout<<weight[0][0][1][0]<<std::endl;
    //    std::cout<<ind[0][0][0][1]<<std::endl;

    torch::Tensor ind_0 = ind.select(3,0).clone().unsqueeze(3);
    torch::Tensor ind_1 = ind.select(3,1).clone().unsqueeze(3);
    //    ind_0.print();
    ind_0 = ind_0.expand({ind.size(0), ind.size(1), ind.size(2), 2});
    ind_1 = ind_1.expand({ind.size(0), ind.size(1), ind.size(2), 2});
    torch::Tensor poly1 = poly.gather(2,ind_0);
    torch::Tensor poly2 = poly.gather(2,ind_1);
    //    poly1.print();
    //    std::cout<<poly1[0][0][0][0]<<std::endl;
    //    std::cout<<poly1[0][0][0][1]<<std::endl;
    //
    //    poly2.print();
    //    std::cout<<poly2[0][0][0][0]<<std::endl;
    //    std::cout<<poly2[0][0][0][1]<<std::endl;
    //    while(1);

    poly = poly1 * (1 - weight) + poly2 * weight;
    //    std::cout<<poly[0][0][0][0]<<std::endl;
    //    std::cout<<poly[0][0][0][1]<<std::endl;
    //    std::cout<<poly[0][0][1][0]<<std::endl;
    //    std::cout<<poly[0][0][1][1]<<std::endl;

    return poly;
}
//[1,100,40,2]
torch::Tensor img_poly_to_can_poly(torch::Tensor img_poly)
{
    //TODO
    /*
          if len(img_poly) == 0:
              return torch.zeros_like(img_poly)
     */

//    img_poly.print();
    auto aaa = img_poly.sizes();
    int len_ = aaa.size();

    torch::Tensor poly_0 = img_poly.select(len_-1,0);
    torch::Tensor poly_1 = img_poly.select(len_-1,1);
    auto x_min_index = torch::min(poly_0,-1);
    auto y_min_index = torch::min(poly_1,-1);
    torch::Tensor x_min = std::get<0>(x_min_index);
    torch::Tensor y_min = std::get<0>(y_min_index);
    torch::Tensor can_poly = img_poly.clone();
    can_poly.select(len_-1,0) = can_poly.select(len_-1,0) - x_min.unsqueeze(-1);//x_min.unsqueeze(2)
    can_poly.select(len_-1,1) = can_poly.select(len_-1,1) - y_min.unsqueeze(-1);//y_min.unsqueeze(2)

    //    std::cout<<can_poly[0][0][1][0]<<std::endl;
    //    std::cout<<can_poly[0][0][1][1]<<std::endl;

    return can_poly;
}

torch::Tensor get_quadrangle(const torch::Tensor box)
{
    torch::Tensor x_min = box.select(2,0);
    torch::Tensor y_min = box.select(2,1);
    torch::Tensor x_max = box.select(2,2);
    torch::Tensor y_max = box.select(2,3);
    //    x_min.print();
    //    y_max.print();

    torch::Tensor tmp = torch::stack({(x_min + x_max) / 2., y_min,
                                      x_min, (y_min + y_max) / 2.,
                                      (x_min + x_max) / 2., y_max,
                                      x_max, (y_min + y_max) / 2.},2);
    tmp = tmp.view({x_min.size(0), x_min.size(1), 4, 2}); //[1, 100, 4, 2]

    //    torch::Tensor aaa = tmp[0][99][1][1];
    //    std::cout<<aaa<<std::endl;

    //    tmp.print();
    //    int a = 0;
    return tmp;

}

torch::Tensor get_init(const torch::Tensor box,string snake_config_init="quadrangle")
{
    if("quadrangle" == snake_config_init)
    {
        return get_quadrangle(box);
    }else
    {
        std::cout<<"get_init not implenet!\n";
        exit(1);
    }
}


// aim [1,10,2,2]   ind_mask_ [1,10] 比如前5个是1余都是0  得到的结果形状是[5,40,2]  即pytorch里面的操作 aim = aim[ind_mask]
torch::Tensor deal_mask_index(torch::Tensor aim_,torch::Tensor ind_mask_)
{
//    aim_.index()
    //    aim_.print();
    torch::Tensor aim = aim_.clone().squeeze(0);//[1,100,40,2]  -->> [100,40,2]
    torch::Tensor ind_mask = ind_mask_.clone().squeeze(0);////[1,100]  -->> [100]
    if(ind_mask.sizes().empty())
    {
        torch::Tensor tmp;
        return tmp;
    }

    int row = ind_mask.size(0);

    int cnt = 0;
    for(int i=0;i<row;i++)
    {
        if(ind_mask[i].item().toInt())
        {
            cnt += 1;
        }
    }
    torch::Tensor out = torch::zeros({cnt,aim.size(1),aim.size(2)});
    int index_ = 0;
    for(int i=0;i<row;i++)
    {
        if(ind_mask[i].item().toInt())
        {
            out[index_++] = aim[i];
            //            std::cout<<i<<std::endl;
        }
    }
    //
    //    std::cout<<"##############################################"<<std::endl;
    //    std::cout<<out<<std::endl;

    return out.to(torch::kCUDA);
}


//gcn_feature[ind == i] = feature
//gcn_feature[5,64,40]      ind_mask_ [5]   feature[5,64,40]
//目的是把ind_mask 为1的块，赋值为feature
void deal_mask_index_assign(torch::Tensor gcn_feature,torch::Tensor ind_mask_,const torch::Tensor feature)
{
    int row = ind_mask_.size(0);

    int cnt = 0;
    for(int i=0;i<row;i++)
    {
        if(ind_mask_[i].item().toInt())//if(ind_mask_[i].cpu().item().toInt())
        {
            gcn_feature[i] = feature[i];
        }
    }
    //    feature.print();
    //    gcn_feature.print();
}

torch::Tensor deal_mask_index_2(torch::Tensor aim_,torch::Tensor ind_mask_)
{
    torch::Tensor aim = aim_.clone().squeeze(0);//[1,100,6]  -->> [100,6]
    torch::Tensor ind_mask = ind_mask_.clone().squeeze(0);////[1,100]  -->> [100]

    int row = ind_mask.size(0);

    int cnt = 0;
    for(int i=0;i<row;i++)
    {
        if(ind_mask[i].item().toInt())
        {
            cnt += 1;
        }
    }
    torch::Tensor out = torch::zeros({cnt,aim.size(1)});
    int index_ = 0;
    for(int i=0;i<row;i++)
    {
        if(ind_mask[i].item().toInt())
        {
            out[index_++] = aim[i];
            //            std::cout<<i<<std::endl;
        }
    }
    //
    //    std::cout<<"##############################################"<<std::endl;
    //    std::cout<<out<<std::endl;

    return out.to(torch::kCUDA);
}

//torch::Tensor deal_mask_index_3(torch::Tensor aim_,torch::Tensor ind_mask_)
//{
//    torch::Tensor aim = aim_.clone().squeeze(0);//[1,100,6]  -->> [100,6]
//    torch::Tensor ind_mask = ind_mask_.clone().squeeze(0);////[1,100]  -->> [100]
//
//    int row = ind_mask.size(0);
//
//    int cnt = 0;
//    for(int i=0;i<row;i++)
//    {
//        if(ind_mask[i].item().toInt())
//        {
//            cnt += 1;
//        }
//    }
//    torch::Tensor out = torch::zeros({cnt,aim.size(1)});
//    int index_ = 0;
//    for(int i=0;i<row;i++)
//    {
//        if(ind_mask[i].item().toInt())
//        {
//            out[index_++] = aim[i];
////            std::cout<<i<<std::endl;
//        }
//    }
////
////    std::cout<<"##############################################"<<std::endl;
////    std::cout<<out<<std::endl;
//
//    return out.to(torch::kCUDA);
//}


//auto output = std::make_tuple(out_ct_hm,out_wh,ct,detection,tmp);
//auto init = std::make_tuple(i_it_4pys,c_it_4pys,ind);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare_testing_init(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor>&in_output)
{
    const torch::Tensor detection = std::get<3>(in_output);//[1,100,6]
    //    detection.print();
    const torch::Tensor detection0_4 = detection.slice(2,0,4);//[1,100,4]
    const torch::Tensor score = detection.slice(2,4,5).squeeze(2);//[1,100]
    //    detection0_4.print();
    //    detection4.print();
    //    print(detection0_4[0][0][0]);
    //    print(detection4[0][0]);

    torch::Tensor i_it_4pys = get_init(detection0_4);
    //    i_it_4pys.print();

    int init_poly_num = 40;
    i_it_4pys = uniform_upsample(i_it_4pys, init_poly_num);
//    i_it_4pys.print();
    torch::Tensor c_it_4pys = img_poly_to_can_poly(i_it_4pys);
    float ct_score = 0.3;
    torch::Tensor ind = score > ct_score;
    //    std::cout<<ind<<std::endl;

    //    i_it_4pys = i_it_4pys[ind]
    i_it_4pys = deal_mask_index(i_it_4pys,ind);
    c_it_4pys = deal_mask_index(c_it_4pys,ind);
    //    c_it_4pys.print();

    //    ind.print();

    std::vector<torch::Tensor> v_tensor;
    for(int i=0;i<ind.size(0);i++)
    {
        torch::Tensor ele = torch::full({ind[i].sum().item().toInt()},i); //  {ind[i].sum().item().toInt()}  这里一定是一维的吗？
        v_tensor.push_back(ele);
    }
    ind = torch::cat(v_tensor,0);

    //    std::cout<<ind<<std::endl;
    //    ind.print();

    //    torch::full()

    //    i_it_4pys.print();
    //    c_it_4pys.print();

    auto init = std::make_tuple(i_it_4pys,c_it_4pys,ind);
    return init;
}



torch::Tensor clip_to_image(torch::Tensor bbox, const int h,const int w)
{
    //    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0)
    //    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1)
    //    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1)
    bbox.select(2,0).clamp_(0,INT_MAX*1.0);
    bbox.select(2,1).clamp_(0,INT_MAX*1.0);

    bbox.select(2,2).clamp_(INT_MIN*1.0,w-1);
    bbox.select(2,3).clamp_(INT_MIN*1.0,h-1);

    return bbox;
}

torch::Tensor nms(torch::Tensor heat, int kernel = 3)
{
    int pad = (kernel-1)/2;
    torch::Tensor hmax = torch::max_pool2d(heat,{kernel,kernel},1,pad);
    //    torch::Tensor aa = hmax[0][0][21][200];
    //    std::cout<<"nms--aa=="<<aa<<std::endl;

    torch::Tensor keep = (hmax == heat).to(torch::kFloat);
    //    torch::Tensor abc = heat * keep;

    //    torch::Tensor aaaaa = abc[0][0][0][0];
    //    std::cout<<"aa----=="<<aaaaa<<std::endl;
    //    int a3 = 0;
    return heat * keep;
}


torch::Tensor gather_feat(torch::Tensor feat, torch::Tensor ind,bool mask = false)
{
    int dim = feat.size(2);
    ind = ind.unsqueeze(2).expand({ind.size(0), ind.size(1), dim});
    feat = feat.gather(1, ind);
    //    feat.print();
    //    torch::Tensor aa = feat[0][2][0];
    //    std::cout<<aa<<std::endl;

    return feat;
}

torch::Tensor transpose_and_gather_feat(torch::Tensor feat, torch::Tensor ind)
{
    feat = feat.permute({0,2,3,1}).contiguous();
    feat = feat.view({feat.size(0), -1, feat.size(3)});
    feat = gather_feat(feat, ind);

    //    torch::Tensor aa = feat[0][56][1];
    //    std::cout<<"now=="<<aa<<std::endl;
    return feat;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor,torch::Tensor> topk(torch::Tensor scores, int K = 40)
{
    int batch_ = scores.size(0);
    int cat_ = scores.size(1);
    int height_ = scores.size(2);
    int width_ = scores.size(3);
    auto tpl = torch::topk(scores.view({batch_,cat_,-1}),K);
    torch::Tensor topk_scores = std::get<0>(tpl);
    //    topk_scores.print();
    torch::Tensor topk_inds = std::get<1>(tpl);
    //    topk_inds.print();


    //    torch::Tensor aaq = topk_scores[0][0][10];
    //    torch::Tensor bb = topk_inds[0][0][32];
    //    std::cout<<"aa="<<aaq<<std::endl;
    //    std::cout<<"bb="<<bb<<std::endl;

    topk_inds = topk_inds % (height_ * width_);
    torch::Tensor cc = topk_inds[0][0][30];
    //     std::cout<<"cc="<<cc<<std::endl;

    torch::Tensor topk_ys = (topk_inds / width_).to(torch::kInt).to(torch::kFloat);
    torch::Tensor topk_xs = (topk_inds % width_).to(torch::kInt).to(torch::kFloat);


    auto tpl_2 = torch::topk(topk_scores.view({batch_,-1}),K);
    torch::Tensor topk_score = std::get<0>(tpl_2);
    torch::Tensor topk_ind = std::get<1>(tpl_2);
    torch::Tensor topk_clses = (topk_ind / K).to(torch::kInt);

    //      torch::Tensor dd = topk_ys[0][0][50];
    //      torch::Tensor ff = topk_xs[0][0][50];
    //      torch::Tensor ee = topk_clses[0][45];
    //      std::cout<<"dd="<<dd<<std::endl;
    //      std::cout<<"ff="<<ff<<std::endl;
    //      std::cout<<"ee="<<ee<<std::endl;

    topk_inds = gather_feat(topk_inds.view({batch_, -1, 1}), topk_ind);
    topk_inds = topk_inds.view({batch_,K});

    topk_ys = gather_feat(topk_ys.view({batch_, -1, 1}), topk_ind);
    topk_ys = topk_ys.view({batch_,K});
    topk_xs = gather_feat(topk_xs.view({batch_, -1, 1}), topk_ind);
    topk_xs = topk_xs.view({batch_,K});

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor,torch::Tensor> mytuple(topk_score, topk_inds, topk_clses, topk_ys, topk_xs);

    //    torch::Tensor  a_ = topk_score[0][78];
    //    torch::Tensor b_ = topk_inds[0][78];
    //    torch::Tensor  c_ = topk_clses[0][79];
    //    torch::Tensor d_ = topk_ys[0][80];
    //    torch::Tensor f_ = topk_xs[0][81];
    //    std::cout<<"a_="<<a_<<std::endl;
    //    std::cout<<"b_="<<b_<<std::endl;
    //    std::cout<<"c_="<<c_<<std::endl;
    //    std::cout<<"d_="<<d_<<std::endl;
    //    std::cout<<"f_="<<f_<<std::endl;

    return mytuple;
}

std::tuple<torch::Tensor, torch::Tensor> decode_ct_hm(torch::Tensor out_ct_hm_sigmod,torch::Tensor wh,bool reg=false,int K=100)
{
    int batch_ = out_ct_hm_sigmod.size(0);
    int cat_ = out_ct_hm_sigmod.size(1);
    int height_ = out_ct_hm_sigmod.size(2);
    int width_ = out_ct_hm_sigmod.size(3);

    torch::Tensor ct_hm = nms(out_ct_hm_sigmod);

    auto my_tuple = topk(ct_hm,100);
    torch::Tensor scores = std::get<0>(my_tuple);
    torch::Tensor inds = std::get<1>(my_tuple);
    torch::Tensor clses = std::get<2>(my_tuple);
    torch::Tensor ys = std::get<3>(my_tuple);
    torch::Tensor xs = std::get<4>(my_tuple);

    wh = transpose_and_gather_feat(wh, inds);
    wh = wh.view({batch_, K, 2});

    if(reg)
    {
        //TODO
    }else
    {
        xs = xs.view({batch_, K, 1});
        ys = ys.view({batch_, K, 1});
    }

    clses = clses.view({batch_, K, 1}).to(torch::kFloat);
    scores = scores.view({batch_, K, 1});
    torch::Tensor ct = torch::cat({xs, ys},2); //pytorch::   ct = torch.cat([xs, ys], dim=2)
    //    ct.print();
    //       ct = torch.cat([xs, ys], dim=2);
    torch::Tensor wh_0 = wh.select(2,0).unsqueeze(2);
    //    wh_0.print();
    //    std::cout<<wh_0[0][0][0]<<std::endl;
    torch::Tensor wh_1 = wh.select(2,1).unsqueeze(2);
    torch::Tensor xs_t0 = xs - wh_0 / 2;
    torch::Tensor ys_t0 = ys - wh_1 / 2;
    torch::Tensor xs_t1 = xs + wh_0 / 2;
    torch::Tensor ys_t1 = ys + wh_1 / 2;
    //    torch::Tensor a123 = xs_t0[0][50][0];
    //    std::cout<<a123<<std::endl;


    //    xs_t0.print();
    //    ys_t0.print();
    //    xs_t1.print();
    //    ys_t1.print();
    //    vector<torch::Tensor> abce = {xs_t0,ys_t0,xs_t1,ys_t1};
    //    torch::Tensor bboxes = torch::cat(abce,2);
    //    std::cout<<"-----cat   shape---"<<std::endl;
    //    bboxes.print();
    //    while(1);

    torch::Tensor bboxes = torch::cat({xs_t0,ys_t0,xs_t1,ys_t1},2);

    //    torch::Tensor aa11 = bboxes[0][23][0];
    //    torch::Tensor aa12 = bboxes[0][23][1];
    //    torch::Tensor  aa13 = bboxes[0][23][2];
    //    torch::Tensor   aa14 = bboxes[0][23][3];
    //    std::cout<<"aa11="<<aa11<<std::endl;
    //    std::cout<<"aa12="<<aa12<<std::endl;
    //    std::cout<<"aa13="<<aa13<<std::endl;
    //    std::cout<<"aa14="<<aa14<<std::endl;
    torch::Tensor detection = torch::cat({bboxes, scores, clses},2);
    //    detection.print();
    //    std::cout<<"-=="<<detection[0][0][0]<<std::endl;

    std::tuple<torch::Tensor, torch::Tensor> mytuple(ct, detection);
    return mytuple;
}

torch::Tensor get_gcn_feature(const torch::Tensor cnn_feature, const torch::Tensor img_poly_, const torch::Tensor ind, const int h, const int w)
{
    // [5, 40, 2]
    torch::Tensor img_poly = img_poly_.clone();

    auto aaa = img_poly.sizes();
    int len_ = aaa.size();
    //    img_poly.print();
    img_poly.select(2,0) = img_poly.select(len_-1,0) / (w / 2.0) - 1;
    img_poly.select(2,1) = img_poly.select(len_-1,1) / (h / 2.0) - 1;
    int batch_size = cnn_feature.size(0);
    //[5,64,40]
    torch::Tensor gcn_feature = torch::zeros({img_poly.size(0), cnn_feature.size(1), img_poly.size(1)}).to(img_poly.device());
    //    gcn_feature.print();

    //torch::Tensor deal_mask_index(torch::Tensor aim_,torch::Tensor ind_mask_)
    for(int i=0;i<batch_size;i++)
    {
        torch::Tensor index_ = (ind == i).to(img_poly.device());
        //        index_.print();
        torch::Tensor tmp = deal_mask_index(img_poly,index_);
//                tmp.print();
        if(0 == tmp.numel()) {continue;}
        torch::Tensor poly = tmp.unsqueeze(0);
        //        poly.print();
        torch::Tensor cnn_feature_i = cnn_feature.slice(0,i,i+1);
        //        cnn_feature.print();
        //        cnn_feature_i.print();

        //[1, 64, 5, 40]
        torch::Tensor abc = torch::grid_sampler_2d(cnn_feature_i,poly,0,0);
        abc = abc[0];
        //[5,64,40]
        torch::Tensor feature = abc.permute({1,0,2});

        deal_mask_index_assign(gcn_feature,index_,feature);
    }

    //    gcn_feature.print();
    //    std::cout<<gcn_feature[0][0][2]<<std::endl;

    return gcn_feature;

}
// i_poly [5,40,2]
//i_poly = i_poly[:, ::10]
//[5,4,2]
torch::Tensor index_maohaomaohao(const torch::Tensor i_poly,const int selete_step)
{
    std::vector<torch::Tensor> v_tmp;
    int size_ = i_poly.size(1);
    int index = 0;
    while (index < size_)
    {
        v_tmp.push_back(i_poly.select(1,index).unsqueeze(1));
        index += selete_step;
    }


    //    for(int i=0;i<v_tmp.size();i++)
    //    {
    //        v_tmp[i].print();
    //    }

    torch::Tensor tmp = torch::cat(v_tmp,1);
    //    tmp.print();
    return tmp;
}

torch::Tensor init_poly(const std::shared_ptr<torch::jit::script::Module> &model_snake0, const std::shared_ptr<torch::jit::script::Module> &model_fuse,const torch::Tensor &cnn_feature,const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> &init)
{
    //    auto init = std::make_tuple(i_it_4pys,c_it_4pys,ind);
    torch::Tensor i_it_4pys = std::get<0>(init);
    torch::Tensor c_it_4pys = std::get<1>(init);
    //    c_it_4pys.print();
    torch::Tensor ind = std::get<2>(init);
    /*TODO
     *         if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)
     * */
    int h = cnn_feature.size(2);
    int w = cnn_feature.size(3);

    //    cnn_feature.print();
    //    i_it_4pys.print();
    //    ind.print();

    torch::Tensor init_feature = get_gcn_feature(cnn_feature, i_it_4pys, ind,h,w);
    //    init_feature.print();

    auto min_tmp = torch::min(i_it_4pys,1);
    auto max_tmp = torch::max(i_it_4pys,1);
    torch::Tensor min_a = std::get<0>(min_tmp);
    torch::Tensor max_a = std::get<0>(max_tmp);
    torch::Tensor center = (min_a + max_a) * 0.5;
    //    center.print();
    //    std::cout<<center<<std::endl;

    //    cnn_feature.print();
    //    center.unsqueeze(1).print();
    //    ind.print();

    torch::Tensor ct_feature = get_gcn_feature(cnn_feature, center.unsqueeze(1), ind,h,w);
    //    ct_feature.print();
    //    std::cout<<ct_feature<<std::endl;

    std::vector<torch::Tensor> v_tensor_tmp = {init_feature,ct_feature.expand_as(init_feature)};
    init_feature = torch::cat(v_tensor_tmp,1);

    init_feature = model_fuse->forward({init_feature}).toTensor();
    //    init_feature.print();
    //    std::cout<<init_feature<<std::endl;
    //    std::cout<<init_feature[0][0][2]<<std::endl;
    //    max_tmp.print();

    std::vector<torch::Tensor> v_tensor_tmp2 = {init_feature,c_it_4pys.permute({0, 2, 1})};
    torch::Tensor init_input = torch::cat(v_tensor_tmp2, 1);
    //    init_input.print();

    int snake_config_adj_num = 4;
    torch::Tensor adj = get_adj_ind(snake_config_adj_num, init_input.size(2), init_input.device());

    torch::Tensor snake_output = model_snake0->forward({init_input,adj}).toTensor();
    //    snake_output.print();
    //    std::cout<<snake_output<<std::endl;

    torch::Tensor i_poly = i_it_4pys + snake_output.permute({0, 2, 1});
    //    std::cout<<i_poly<<std::endl;

    int snake_config_init_poly_num = 40;
    int dim_selete_step = snake_config_init_poly_num / 4;
    //    torch::Tensor i_poly_slice = i_poly.slice(1,0,dim_end);
    //    std::cout<<i_poly_slice<<std::endl;

    i_poly = index_maohaomaohao(i_poly,dim_selete_step);
    //    std::cout<<i_poly<<std::endl;




    //    float snake_config_ro = 4.0;
    //    c_it_4pys = c_it_4pys * snake_config_ro;
    //    std::vector<torch::Tensor> v_tmp = {init_feature, c_it_4pys.permute({0, 2, 1})};
    //    torch::Tensor init_input = torch::cat(v_tmp, 1);
    //    init_input.print();

    //    torch::Tensor evolve;
    return i_poly;
}

//void test_tensor(torch::Tensor a)
//{
//    a[0][0] = -100;
//
//}

//// aim [1,10,2,2]   ind_mask_ [1,10] 比如前5个是1余都是0  得到的结果形状是[5,40,2]  即pytorch里面的操作 aim = aim[ind_mask]
//torch::Tensor deal_mask_index11(torch::Tensor aim_,torch::Tensor ind_mask_)
//{
//    torch::Tensor aim = aim_.clone().squeeze(0);//[1,100,40,2]  -->> [100,40,2]
//    torch::Tensor ind_mask = ind_mask_.clone().squeeze(0);////[1,100]  -->> [100]
////    std::cout<<ind_mask<<std::endl;
//    ind_mask.unsqueeze_(1).unsqueeze_(2);
//    ind_mask = ind_mask.expand({aim.size(0), aim.size(1), aim.size(2)});
////    ind_0 = ind_0.expand({ind.size(0), ind.size(1), ind.size(2), 2});
////    std::cout<<"##############################################"<<std::endl;
////    std::cout<<ind_mask<<std::endl;
//    torch::Tensor tmp = ind_mask * aim;
//    std::cout<<aim<<std::endl;
//    std::cout<<"##############################################"<<std::endl;
//    std::cout<<tmp<<std::endl;
//
//
//
//    return ind_mask;
//}
//
//// aim [1,10,2,2]   ind_mask_ [1,10] 比如前5个是1余都是0  得到的结果形状是[5,40,2]  即pytorch里面的操作 aim = aim[ind_mask]
//torch::Tensor deal_mask_index22(torch::Tensor aim_,torch::Tensor ind_mask_)
//{
//    torch::Tensor aim = aim_.clone().squeeze(0);//[1,100,40,2]  -->> [100,40,2]
//    torch::Tensor ind_mask = ind_mask_.clone().squeeze(0);////[1,100]  -->> [100]
//
//    int row = ind_mask.size(0);
//
//    int cnt = 0;
//    for(int i=0;i<row;i++)
//    {
//        if(ind_mask[i].item().toInt())
//        {
//            cnt += 1;
//        }
//    }
//    torch::Tensor out = torch::zeros({cnt,aim.size(1),aim.size(2)});
//    int index_ = 0;
//    for(int i=0;i<row;i++)
//    {
//        if(ind_mask[i].item().toInt())
//        {
//            out[index_++] = aim[i];
////            std::cout<<i<<std::endl;
//        }
//    }
////
////    std::cout<<"##############################################"<<std::endl;
////    std::cout<<out<<std::endl;
//
//    return out;
//}

//ex shape[1,5,4,2]      ex[..., 0, 1] -->>[1,5]
torch::Tensor index_tensor_3(const torch::Tensor ex,const int idx1,const int idx2)
{
//    ex.print();
    int dim_ = ex.size(1);
    torch::Tensor out = torch::empty({1,dim_}).to(ex.device());
    int size_ = ex.size(1);
    for(int i=0;i<size_;i++)
    {
        auto a = ex[0][i][idx1][idx2];
        out[0][i] = a;
        //        std::cout<<a<<std::endl;
    }


    return out;
}

//ex shape[1,5,4,2]
torch::Tensor get_octagon(const torch::Tensor ex)
{
    // w, h = ex[..., 3, 0] - ex[..., 1, 0], ex[..., 2, 1] - ex[..., 0, 1]
    torch::Tensor w = index_tensor_3(ex,3,0) - index_tensor_3(ex,1,0);
    torch::Tensor h = index_tensor_3(ex,2,1)- index_tensor_3(ex,0,1);

    torch::Tensor t = index_tensor_3(ex,0,1);
    torch::Tensor l = index_tensor_3(ex,1,0);
    torch::Tensor b = index_tensor_3(ex,2,1);
    torch::Tensor r = index_tensor_3(ex,3,0);
    float x = 8.0;
    //    t, l, b, r = ex[..., 0, 1], ex[..., 1, 0], ex[..., 2, 1], ex[..., 3, 0]
    //        x = 8.

    //    std::cout<<w<<std::endl;
    //    std::cout<<h<<std::endl;
    //     std::cout<<r<<std::endl;
    //    torch::Tensor abc = torch::max(index_tensor_3(ex,0,0) - w / x, l);
    //     std::cout<<abc<<std::endl;
    std::vector<torch::Tensor> octagon_ = {
        index_tensor_3(ex,0,0), index_tensor_3(ex,0,1),
        torch::max(index_tensor_3(ex,0,0) - w / x, l), index_tensor_3(ex,0,1),
        index_tensor_3(ex,1,0), torch::max(index_tensor_3(ex,1,1) - h / x, t),
        index_tensor_3(ex,1,0), index_tensor_3(ex,1,1),
        index_tensor_3(ex,1,0), torch::min(index_tensor_3(ex,1,1) + h / x, b),
        torch::max(index_tensor_3(ex,2,0) - w / x, l), index_tensor_3(ex,2,1),
        index_tensor_3(ex,2,0), index_tensor_3(ex,2,1),
        torch::min(index_tensor_3(ex,2,0) + w / x, r), index_tensor_3(ex,2,1),
        index_tensor_3(ex,3,0), torch::min(index_tensor_3(ex,3,1) + h / x, b),
        index_tensor_3(ex,3,0), index_tensor_3(ex,3,1),
        index_tensor_3(ex,3,0), torch::max(index_tensor_3(ex,3,1) - h / x, t),
        torch::min(index_tensor_3(ex,0,0) + w / x, r), index_tensor_3(ex,0,1)
    };

    torch::Tensor octagon = torch::stack(octagon_,2).contiguous().view({t.size(0),t.size(1),12,2});;

    //    octagon.print();
    //    std::cout<<octagon<<std::endl;
    return octagon;

}

std::tuple<torch::Tensor,torch::Tensor> prepare_testing_evolve_1(torch::Tensor ex)
{
    // TODO:  if len(ex) == 0:

    torch::Tensor i_it_pys = get_octagon(ex.unsqueeze(0));
    int snake_config_poly_num = 128;
    i_it_pys = uniform_upsample(i_it_pys, snake_config_poly_num);
    i_it_pys = i_it_pys.select(0,0);
                                              //[5,128,2]
    torch::Tensor c_it_pys = img_poly_to_can_poly(i_it_pys);
//    std::cout<<c_it_pys<<std::endl;
//    c_it_pys.print();

    // std::tuple<torch::Tensor,torch::Tensor> evolve = std::make_tuple(i_it_pys,c_it_pys);
    std::tuple<torch::Tensor,torch::Tensor> evolve = std::make_tuple(i_it_pys,c_it_pys);
    return evolve;
}

std::tuple<torch::Tensor,torch::Tensor> prepare_testing_evolve(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&output2,
                                     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>&output3,
                                     const int h, const int w)
{
    std::get<0>(output3) = std::get<0>(output2);
    std::get<1>(output3) = std::get<1>(output2);
    std::get<2>(output3) = std::get<2>(output2);
    std::get<3>(output3) = std::get<3>(output2);
    std::get<4>(output3) = std::get<4>(output2);
    std::get<5>(output3) = std::get<5>(output2);
    torch::Tensor ex_ = std::get<5>(output3);//[5,4,2]
    ex_.select(2,0).clamp_(0,w-1);
    ex_.select(2,1).clamp_(0,h-1);
    std::get<5>(output3) = ex_;
    //    ex_.print();
    //    torch::Tensor aa = index_tensor_3(ex_.unsqueeze(0),0,1);
    //    std::cout<<aa<<std::endl;


    // std::tuple<torch::Tensor,torch::Tensor> evolve = std::make_tuple(i_it_pys,c_it_pys);
    std::tuple<torch::Tensor,torch::Tensor> evolve = prepare_testing_evolve_1(ex_);
    std::get<6>(output3) = std::get<0>(evolve);//output.update({'it_py': evolve['i_it_py']})

    return evolve;
}

torch::Tensor evolve_poly(const std::shared_ptr<torch::jit::script::Module> &m_model_evolve_gcn, const torch::Tensor &cnn_feature, const torch::Tensor &i_it_py_evolve, torch::Tensor &c_it_py_evolve, const torch::Tensor &ind_init)
{
//    c_it_py_evolve[0][0][0] = -1000;
    /* TODO:
     *  if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
*/
    int h = cnn_feature.size(2);
    int w = cnn_feature.size(3);
    torch::Tensor init_feature = get_gcn_feature(cnn_feature, i_it_py_evolve, ind_init, h, w);
//    init_feature.print();
//    std::cout<<init_feature<<std::endl;

    float snake_config_ro = 4, snake_config_adj_num=4;
    c_it_py_evolve = c_it_py_evolve * snake_config_ro;
    std::vector<torch::Tensor> v_tmp = {init_feature, c_it_py_evolve.permute({0, 2, 1})};
    torch::Tensor init_input = torch::cat(v_tmp, 1);
    torch::Tensor adj = get_adj_ind(snake_config_adj_num, init_input.size(2), init_input.device());
//     std::cout<<adj<<std::endl;
//    adj.print();

    torch::Tensor snake_output = m_model_evolve_gcn->forward({init_input,adj}).toTensor();
    torch::Tensor i_poly = i_it_py_evolve * snake_config_ro + snake_output.permute({0, 2, 1});
//    std::cout<<i_poly<<std::endl;
//    i_poly.print();

    return i_poly;
}

void visualize_training_box(const cv::Mat &img_,const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,std::vector<torch::Tensor>> &outpu4)
{
    std::cout<<"img_size_"<<img_.size()<<std::endl;
    cv::Mat img = img_.clone();
    float snake_config_down_ratio = 4.0;
    //这里outpu4里面有7个东西，分别是ct_hm,wh,ct,detection,it_ex, ex, it_py pys
    torch::Tensor box = std::get<3>(outpu4); // box [5,6]
//    box.print();
//    std::cout<<box<<std::endl;
//    box = box.slice(1,0,4).detach().cpu();//[5,4]
//    box.print();
    box.slice(1,0,4) = box.slice(1,0,4) * snake_config_down_ratio;
    box = box.detach().cpu();//[5,6]

    torch::Tensor ex = std::get<7>(outpu4)[2];
    ex = ex * snake_config_down_ratio;
    ex = ex.detach().cpu();//[5,128,2]

    auto result_data = box.accessor<float, 2>();
    auto ex_data = ex.accessor<float, 3>();
    int dim = ex_data.size(1);

    std::vector<std::vector<cv::Point>> vv_pt;
    for(int i=0;i<result_data.size(0);i++)
    {
        float score = result_data[i][4];
        int id_label = result_data[i][5];
//        std::cout<<"id_label="<<id_label<<"      score="<<score<<std::endl;
//        if(score < 0.4) { continue;}
        int xmin = result_data[i][0];
        int ymin = result_data[i][1];
        int xmax = result_data[i][2];
        int ymax = result_data[i][3];
        std::vector<cv::Point> v_pt;
        for(int j=0;j<dim;j++)
        {
            int x = ex_data[i][j][0];
            int y = ex_data[i][j][1];
            v_pt.push_back(cv::Point(x,y));
        }
        vv_pt.push_back(v_pt);

        float ratio = xmin * 1.0 / img.cols;
        std::cout<<"xmin="<<xmin<<std::endl;
        std::cout<<"ratio=="<<ratio<<std::endl;

        cv::rectangle(img,cv::Point(xmin,ymin),cv::Point(xmax,ymax),cv::Scalar(255,0,0),1);
//        cv::putText(img,label_map[id_label],cv::Point(x1,y2),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,55));
    }
//    cv::drawContours(img,vv_pt,-1,cv::Scalar(0,255,255),2);

//    cv::namedWindow("show",0);
    cv::imshow("show",img);
    cv::waitKey(0);
}

void box2src(const cv::Mat &m_draw, const cv::Mat &m_src, const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,std::vector<torch::Tensor>> &outpu4)
{
    std::cout<<"box2src m_draw_size_"<<m_draw.size()<<std::endl;
    cv::Mat img = m_src.clone();
    float snake_config_down_ratio = 4.0;
    //这里outpu4里面有7个东西，分别是ct_hm,wh,ct,detection,it_ex, ex, it_py pys
    torch::Tensor box = std::get<3>(outpu4); // box [5,6]
//    box.print();
//    std::cout<<box<<std::endl;
//    box = box.slice(1,0,4).detach().cpu();//[5,4]
//    box.print();
    box.slice(1,0,4) = box.slice(1,0,4) * snake_config_down_ratio;
    box = box.detach().cpu();//[5,6]

    torch::Tensor ex = std::get<7>(outpu4)[2];
    ex = ex * snake_config_down_ratio;
    ex = ex.detach().cpu();//[5,128,2]

    auto result_data = box.accessor<float, 2>();
    auto ex_data = ex.accessor<float, 3>();
    int dim = ex_data.size(1);

    std::vector<std::vector<cv::Point>> vv_pt;
    for(int i=0;i<result_data.size(0);i++)
    {
        float score = result_data[i][4];
        int id_label = result_data[i][5];
//        std::cout<<"id_label="<<id_label<<"      score="<<score<<std::endl;
//        if(score < 0.4) { continue;}
        int xmin = result_data[i][0];
        int ymin = result_data[i][1];
        int xmax = result_data[i][2];
        int ymax = result_data[i][3];
//        std::cout<<"xmin="<<xmin<<"  cols="<<m_draw.cols<<std::endl;
//        std::cout<<"ymax="<<ymax<<"  rows="<<m_draw.rows<<std::endl;

//        float ratio = xmin * 1.0 / m_draw.cols;
//        std::cout<<"ratio=="<<ratio<<std::endl;

        float ratio = xmin * 1.0 / m_draw.cols;
//        std::cout<<"xmin="<<xmin<<std::endl;
//        std::cout<<"ratio=="<<ratio<<std::endl;

        xmin = xmin * 1.0 / m_draw.cols * m_src.cols;
        xmax = xmax * 1.0 / m_draw.cols * m_src.cols;
        ymin = ymin * 1.0 / m_draw.rows * m_src.rows;
        ymax = ymax * 1.0 / m_draw.rows * m_src.rows;


        std::vector<cv::Point> v_pt;
        for(int j=0;j<dim;j++)
        {
            int x = ex_data[i][j][0];
            int y = ex_data[i][j][1];
            x =  x * 1.0 / m_draw.cols * m_src.cols;
            y = y * 1.0 / m_draw.rows * m_src.rows;
            v_pt.push_back(cv::Point(x,y));
        }
        vv_pt.push_back(v_pt);
//        std::cout<<"xmin="<<xmin<<std::endl;
//        std::cout<<"ymin="<<ymin<<std::endl;
        cv::rectangle(img,cv::Point(xmin,ymin),cv::Point(xmax,ymax),cv::Scalar(0,255,255),2);
//        cv::putText(img,"he",cv::Point(xmin,ymin),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,55));

//        cv::imshow("show-2",img);
//        cv::waitKey(0);
//        int a = 0;
//        cv::putText(img,label_map[id_label],cv::Point(x1,y2),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,55));
    }
    cv::drawContours(img,vv_pt,-1,cv::Scalar(255,0,0),2);

//    cv::namedWindow("show-src",0);
    cv::imshow("show-src",img);
    cv::waitKey(0);
}

torch::Tensor img2tensor(const string img_path,cv::Mat &img_show,cv::Mat &img_src)
{
    Mat img = imread(img_path,IMREAD_UNCHANGED);
    img_src = img.clone();
    int width = img.cols;
    int height = img.rows;
    int x = 32;
    int input_w = (int(width / 1.) | (x - 1)) + 1;
    int input_h = (int(height / 1.) | (x - 1)) + 1;
    cv::resize(img,img,cv::Size(input_w,input_h));
    img_show = img.clone();
//    cv::resize(img,img,cv::Size(1024,1024));

//    img = imread("/data_2/project_202009/libtorch/snake_libtorch_cuda8/99-22.png");


    Mat img2;
    img.convertTo(img2, CV_32F);
    img2 = img2 / 255.0;
    Mat m_out_2 = img2 - cv::Scalar(0.40789655,0.44719303,0.47026116);
    vector<float> v_std_ = {0.2886383,0.27408165,0.27809834};

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(m_out_2, bgrChannels);
    for(int i=0;i<3;i++)
    {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / v_std_[i]);
    }
    Mat m_out_3;
    cv::merge(bgrChannels, m_out_3);


    torch::Tensor input_tensor = torch::from_blob(
            m_out_3.data, {m_out_3.rows, m_out_3.cols, 3}).toType(torch::kFloat32);//torch::kByte //大坑
    //[3,320,320]
    input_tensor = input_tensor.permute({2,0,1});
    //    input_tensor.print();
    //    std::cout<<input_tensor[0][224][250]<<std::endl;

    input_tensor = input_tensor.unsqueeze(0);
    input_tensor = input_tensor.to(torch::kFloat).to(torch::kCUDA);
    return input_tensor;
}



int main(int argc, const char* argv[])
{
    void* handle = dlopen("libdcn_v2_cuda_forward_v2.so", RTLD_LAZY);
    //     void* handle1 = dlopen("libextreme_utils.so", RTLD_LAZY);


    std::cout<<"~~~load model ing~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    string path_pt_centernet ="/data_2/snake-master-cuda8/myfile/pt/CenterNet.pt";
    torch::Device m_device(torch::kCUDA);
    std::shared_ptr<torch::jit::script::Module> m_model_center = torch::jit::load(path_pt_centernet);
    m_model_center->to(m_device);

    string path_pt_snake0 ="/data_2/snake-master-cuda8/myfile/pt/snake_init.pt";
    std::shared_ptr<torch::jit::script::Module> m_model_snake_init = torch::jit::load(path_pt_snake0);
    m_model_snake_init->to(m_device);

    string path_snake_self ="/data_2/snake-master-cuda8/myfile/pt/snake_self.pt";
    std::shared_ptr<torch::jit::script::Module> m_snake_self = torch::jit::load(path_snake_self);
    m_snake_self->to(m_device);

    string path_pt_fuse ="/data_2/snake-master-cuda8/myfile/pt/fuse.pt";
    std::shared_ptr<torch::jit::script::Module> m_model_fuse = torch::jit::load(path_pt_fuse);
    m_model_fuse->to(m_device);

    string path_snake_iter0 ="/data_2/snake-master-cuda8/myfile/pt/snake_iter0.pt";
    std::shared_ptr<torch::jit::script::Module> m_snake_iter0 = torch::jit::load(path_snake_iter0);
    m_snake_iter0->to(m_device);

    string path_snake_iter1 ="/data_2/snake-master-cuda8/myfile/pt/snake_iter1.pt";
    std::shared_ptr<torch::jit::script::Module> m_snake_iter1 = torch::jit::load(path_snake_iter1);
    m_snake_iter1->to(m_device);
    //    m_model_fuse->eval()
    std::cout<<"~~~load model ok!!!~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;

    fstream infile("/data_2/snake/test_pic/list.txt");
    string path_img;
    int cnt_ = 0;
    while(infile >> path_img)
    {
        std::cout<<cnt_++<<"   path="<<path_img<<std::endl;
        Mat img_show,img_src;
        torch::Tensor input_tensor = img2tensor(path_img,img_show,img_src);

        auto out = m_model_center->forward({input_tensor});
        auto tpl = out.toTuple();
        auto out_ct_hm = tpl->elements()[0].toTensor();
//        out_ct_hm.print();
//        std::cout<<out_ct_hm<<std::endl;
        auto out_wh = tpl->elements()[1].toTensor();
//    std::cout<<out_wh<<std::endl;
//        out_wh.print();
        auto out_cnn_feature = tpl->elements()[2].toTensor();
//        out_cnn_feature.print();



        torch::Tensor out_ct_hm_sigmod = torch::sigmoid(out_ct_hm);
        //    auto a1 = out_ct_hm_sigmod[0][0][20][22];
        //    std::cout<<"a1====="<<a1<<std::endl;

        std::tuple<torch::Tensor, torch::Tensor> my_tuple = decode_ct_hm(out_ct_hm_sigmod,out_wh);
        torch::Tensor ct = std::get<0>(my_tuple);
        torch::Tensor detection = std::get<1>(my_tuple);
        detection = clip_to_image(detection, out_cnn_feature.size(2), out_cnn_feature.size(3));
        //    torch::Tensor aa1 = ct[0][88][0];
        //    torch::Tensor aa2 = ct[0][8][1];
        //    torch::Tensor bb1 = detection[0][21][0];
        //    torch::Tensor bb2 = detection[0][21][1];
        //    torch::Tensor bb3 = detection[0][21][2];
        //    torch::Tensor bb4 = detection[0][21][3];
        //    torch::Tensor bb5 = detection[0][21][4];
        //    torch::Tensor bb6 = detection[0][21][5];
        //    std::cout<<"aa1="<<aa1<<std::endl;
        //    std::cout<<"aa2="<<aa2<<std::endl;
        //    std::cout<<"bb1="<<bb1<<std::endl;
        //    std::cout<<"bb2="<<bb2<<std::endl;
        //    std::cout<<"bb3="<<bb3<<std::endl;
        //    std::cout<<"bb4="<<bb4<<std::endl;
        //    std::cout<<"bb5="<<bb5<<std::endl;
        //    std::cout<<"bb6="<<bb6<<std::endl;


        //以上，执行完了centernet及其后处理部分
        //output = self.gcn(output, cnn_feature, batch)
        //这里，输入的output里面有5个东西，分别是ct_hm,wh,ct,detection,it_ex
        torch::Tensor it_ex_tmp;
        auto output = std::make_tuple(out_ct_hm,out_wh,ct,detection,it_ex_tmp);

        //auto init = std::make_tuple(i_it_4pys,c_it_4pys,ind);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> init = prepare_testing_init(output);

        torch::Tensor mask_1 = detection.slice(2,4,5).squeeze(2).clone();//[1,100]
        const float snake_config_ct_score = 0.3;
        mask_1 = mask_1 > snake_config_ct_score;
        torch::Tensor detection_new = deal_mask_index_2(detection,mask_1);
        //    detection_new.print();
        std::get<3>(output) = detection_new;
        std::get<4>(output) = std::get<0>(init);

        /*
         * 此处完成了 init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
         * 得到输出init //auto init = std::make_tuple(i_it_4pys,c_it_4pys,ind);
         * 更新了output std::make_tuple(out_ct_hm,out_wh,ct,detection,it_ex);   //output.update({'it_ex': init['i_it_4py']})
         * 注意后续用it_ex 代表 i_it_4py
        */


        //    save_tensor_txt(out_cnn_feature,"/data_1/everyday/0914/cnnfeature_libtorch.txt");
        //    while (1);

        //        cnn_feature_i.print();
        //    std::cout<<out_cnn_feature<<std::endl;
        //    out_cnn_feature.print();
        //    print(out_cnn_feature[0][63][175][231]);
        //    print(out_cnn_feature[0][63][175][226]);
        //    print(out_cnn_feature[0][63][175][225]);
        //    print(out_cnn_feature[0][63][175][224]);
        //    print(out_cnn_feature[0][63][175][223]);
        //    print(out_cnn_feature[0][63][175][222]);
        //    print(out_cnn_feature[0][63][175][221]);
        //    print(out_cnn_feature[0][63][175][220]);
        //    print(out_cnn_feature[0][63][175][219]);
        //    print(out_cnn_feature[0][63][175][218]);
        //    print(out_cnn_feature[0][63][175][217]);
        //    print(out_cnn_feature[0][63][175][216]);
        //    print(out_cnn_feature[0][63][175][215]);



        torch::Tensor ex = init_poly(m_model_snake_init,m_model_fuse,out_cnn_feature,init);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> outpu2 = \
            std::tuple_cat(output, std::make_tuple(ex));
        /*此处完成对应的pytorch如下两句：
         *  ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                    ret.update({'ex': ex})

           这里outpu2里面有6个东西，分别是ct_hm,wh,ct,detection,it_ex, ex
         * */

        //这里outpu3里面有7个东西，分别是ct_hm,wh,ct,detection,it_ex, ex, it_py
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> output3;

// std::tuple<torch::Tensor,torch::Tensor> evolve = std::make_tuple(i_it_pys,c_it_pys);
        std::tuple<torch::Tensor,torch::Tensor> evolve = prepare_testing_evolve(outpu2, output3, out_cnn_feature.size(2), out_cnn_feature.size(3));
//    std::get<1>(evolve).print();
//    std::cout<< std::get<1>(evolve)[0][0][0]<<std::endl;

        torch::Tensor py = evolve_poly(m_snake_self, out_cnn_feature, std::get<0>(evolve), std::get<1>(evolve), std::get<2>(init));
//     std::cout<< std::get<1>(evolve)[0][0][0]<<std::endl;

        float snake_config_ro = 4.0;
        std::vector<torch::Tensor> pys = {py/snake_config_ro};


        //iter 0
        py = py / snake_config_ro;
        torch::Tensor c_py = img_poly_to_can_poly(py);
        py = evolve_poly(m_snake_iter0, out_cnn_feature, py, c_py, std::get<2>(init));
        pys.push_back(py/snake_config_ro);
//    std::cout<<py<<std::endl;
//    py.print();

        //iter 1
        py = py / snake_config_ro;
        c_py = img_poly_to_can_poly(py);
        py = evolve_poly(m_snake_iter1, out_cnn_feature, py, c_py, std::get<2>(init));
        pys.push_back(py/snake_config_ro);
//
//    std::cout<<py<<std::endl;
//    py.print();


        //这里outpu3里面有7个东西，分别是ct_hm,wh,ct,detection,it_ex, ex, it_py
        //这里outpu4里面有8个东西，分别是ct_hm,wh,ct,detection,it_ex, ex, it_py pys
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,std::vector<torch::Tensor>> outpu4 = \
            std::tuple_cat(output3, std::make_tuple(pys));

        Mat m_show = cv::imread(path_img,-1);
//        visualize_training_box(img_show,outpu4);

        box2src(img_show,img_src,outpu4);

    }



    return 0;
}

