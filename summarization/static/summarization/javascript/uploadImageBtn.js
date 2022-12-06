console.log("uploadImageBtn.js");

$(document).on('click', '.upload-portfolio-image-btn', function() {
    var inputToUploadPortfolioImage = $(this).next('.upload-portfolio-image')
    var portfolioImageGroup = inputToUploadPortfolioImage.closest('.form-row').next().next('.portfolio-image-group')
    var uploadImg = document.getElementById('uploaded-img');
    var resultElement = document.getElementById('uploaded-img-path');
    inputToUploadPortfolioImage.click()
    inputToUploadPortfolioImage.off('change').on('change', function(e) {
      if (e.target.files && e.target.files[0]) {
        var files = e.target.files
        for (var i = 0; i < files.length; i++) {
          var file = files[i]
          var reader = new FileReader()
          reader.onload = function (e) {
            uploadImg.src = e.target.result;
            // console.log(file);
            // resultElement.childNodes[0].nodeValue = file.name;
            resultElement.appendChild( new Text( file.name ) ) ;
            uploadImg.style = " width: 100%; object-fit: cover; border-radius: 10px 10px 10px 10px; border: 6px solid #bababa; border-color: rgb(131, 131, 131);"
            // portfolioImageGroup.append(`
            //   <div class="form-group col-md-3">
            //     <div class="responsive-img-wrapper portfolio-image">
            //       <div class="responsive-img" style="background-image: url(${e.target.result}) width:100% height:100%">
            //       <img id="uploaded-img" width="200" src=${e.target.result}>
            //       </div>
            //     </div>
            //   </div>`)
          }
          reader.readAsDataURL(file)
        }
      }
    })
  })