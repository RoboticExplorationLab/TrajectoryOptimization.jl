_path = "C:\\Users\\taylor howell\\Desktop\\quad_maze_frames"
using FileIO
using ImageMagick
using Images, ImageView

anim = @animate for _img in readdir(_path)
    im_ = FileIO.load(joinpath(_path,_img))
    im_crop = im_[250:end-50,400:end-400]

    plot(im_crop,axis=false)
end
gif(anim, joinpath(_path,"maze_v2.gif"), fps = 20)
