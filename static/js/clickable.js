  // Load folder and file names into the dropdown menus
  var folder_list = null;
  var file_list = null;
  var patch_file_list = null;
  var foldername = null;
  var filename = null;
  var patch_filename = null;
  var img_name = ""; 
  const parent_dir = './static/clickable_results/';
  const test_epoch = 'test_best';
  var coords_url = null;
  const img_size = 1536;
  const full_image_postfix = '_masked_full.png';
  // Define the points for clicking 
  var coords_data = [{
    x: [0, 200, 448],
    y: [0, 200, 340],
    mode: 'markers',
    type: 'scatter',
      marker: {
          size: 8,
          color: 'rgba(30, 230, 216 ,1)',}
    }];
  var clicked_index = 0; 
  var layout = {
      margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0,
      pad: 0
    },
    title: false,
    images: [
        {
          "source":  parent_dir + foldername + "/" + test_epoch + "/images/" + filename + "/" + img_name.replace('.png', full_image_postfix),
          "xref": "x",
          "yref": "y",
          "x": 0,
          "y": 0,
          "sizex": img_size,
          "sizey": img_size,
          "xanchor": "left",
          "yanchor": "bottom",
          "sizing": "stretch",
          "layer":"below",
        },
      ],
      xaxis: {
      showgrid: false,
      showline: false,
      showticklabels: false,
      domain: [0, img_size],
      range: [0, img_size],
      fixedrange: true,
      zeroline: false,
      },
      yaxis: {
      scaleanchor: "x",
      showgrid: false,
      showline: false,
      showticklabels: false,
      domain: [0, img_size],
      range: [0, img_size],
      fixedrange: true,
      zeroline: false,
      }
  };
  var config = {
      displayModeBar: false,
  }
  
  const leftPlot = document.getElementById('div_left');
  // const patch_left = document.getElementById('patch_left');
  const real_visual = document.getElementById('real_visual');
  const real_gx = document.getElementById('real_gx');
  const real_gy = document.getElementById('real_gy');
  const real_normal = document.getElementById('real_normal');
  
  const fake_visual = document.getElementById('fake_visual');
  const fake_gx = document.getElementById('fake_gx');
  const fake_gy = document.getElementById('fake_gy');
  const fake_normal = document.getElementById('fake_normal');
  
  fig_left = Plotly.newPlot('div_left', coords_data, layout, config);
  
  async function fetchFolderData(url){
      let response = await fetch(url);
      let jsondata = await response.json();
      jsondata = JSON.stringify(jsondata);
      jsondata = JSON.parse(jsondata);
      folders = jsondata.folders;
      files = jsondata.files;
      patch_files = jsondata.patch_files;
      onLoadFolderData(folders, files, patch_files);
  };
  const folders_url = parent_dir + "clickable_folders_data.json";
  fetchFolderData(folders_url);
  
  function onLoadFolderData(folders, files, patch_files){
      folder_list = folders;
      file_list = files;
      patch_file_list = patch_files;
  
      // set the options for each dropdown menu 
      for (var i = 0; i < folder_list.length; i++) {
          var opt = document.createElement('option');
          opt.value = folder_list[i];
          opt.innerHTML = folder_list[i];
          document.getElementById('folders').appendChild(opt);
      }
      for (var i = 0; i < file_list.length; i++) {
          var opt = document.createElement('option');
          opt.value = file_list[i];
          opt.innerHTML = file_list[i];
          document.getElementById('files').appendChild(opt);
      }
  
      foldername = folder_list[0];
      filename = file_list[0];
      patch_filename = patch_file_list[0];
      img_name = foldername.split("_")[0] + "_test_0_padded_1800_edge.png";
      coords_url = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_patch_coords.json";
      fetchCoordsData(coords_url);
  };
  
  async function fetchCoordsData(url){
      let response = await fetch(url);
      let jsondata = await response.json();
      jsondata = JSON.stringify(jsondata);
      jsondata = JSON.parse(jsondata);
      coords = jsondata.coords;
      onLoadComplete(coords);
  };
  
  function update_patch_img_path(){
    // patch_left.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_patch_" + clicked_index + "/" + img_name;
    real_visual.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_real_I_patch_" + clicked_index + "/" + img_name;
    real_gx.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_real_gx_patch_" + clicked_index + "/" + img_name;
    real_gy.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_real_gy_patch_" + clicked_index + "/" + img_name;
    real_normal.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_real_normal_patch_" + clicked_index + "/" + img_name;
    fake_visual.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_fake_I_patch_" + clicked_index + "/" + img_name;
    fake_gx.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_fake_gx_patch_" + clicked_index + "/" + img_name;
    fake_gy.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_fake_gy_patch_" + clicked_index + "/" + img_name;
    fake_normal.src = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_fake_normal_patch_" + clicked_index + "/" + img_name;
  };
  
  function onLoadComplete(coords) {
      coords_data[0].x = coords.x;
      coords_data[0].y = coords.y;
      layout.images[0].source = parent_dir + foldername + "/" + test_epoch + "/images/" + filename + "/"  + img_name.replace('.png', full_image_postfix);
      Plotly.redraw('div_left', coords_data, layout, config);
      update_patch_img_path(); // initialize with the first patch
      leftPlot.on('plotly_click', function(pts){
      clicked_index = pts.points[0].pointNumber;
      console.log("click index " + clicked_index);
      update_patch_img_path();
      });
  
  };
  
  // Change function for dropdown list, update the image plot
  $('#folders').change(function() {
  console.log('folder event');
      foldername = $('#folders').val();
      img_name = foldername.split("_")[0] + "_test_0_padded_1800_edge.png"; 
      coords_url = parent_dir + foldername + "/" + test_epoch + "/images/" + "test_patch_coords.json";
      layout.images[0].source = parent_dir + foldername + "/" + test_epoch + "/images/" + filename + "/"  + img_name.replace('.png', full_image_postfix);
      fetchCoordsData(coords_url);
      update_patch_img_path();
  });
  

  $('#files').change(function() {
  console.log('files event');
      filename = $('#files').val();
      layout.images[0].source = parent_dir + foldername  + "/" + test_epoch + "/images/" + filename + "/" + img_name.replace('.png', full_image_postfix);
      Plotly.redraw('div_left', coords_data, layout, config);
      update_patch_img_path();
  });
  