console.log("cancelBtn.js");

$(document).on('click', '#btn-to-cancel-upload', function() {
    var form = document.getElementById('id-img');
    var uploadImg = document.getElementById('uploaded-img');
    var resultElement = document.getElementById('uploaded-img-path');
    uploadImg.src = "";
    console.log(resultElement.childNodes);
    resultElement.removeChild(resultElement.childNodes[0]);
    // resultElement.childNodes[0].nodeValue = "";
    uploadImg.style = "";
    $("#btn-to-submit").prop("disabled", true);
    // submitBtn.prop("disabled", true);
  })

$(document).on('click', '#btn-to-submit', function() {
    $('#form-container').hide();
    $('.form-btn').hide();
    $('#loader').show();
  })