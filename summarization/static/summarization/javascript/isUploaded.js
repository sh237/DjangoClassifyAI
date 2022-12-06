const inputFile = document.getElementById("id_image");
function openFile() {
  document.body.onfocus = getEvent;
  console.log("isUploaded.js");
}
// console.log("isUploaded.js");
function getEvent() {
  setTimeout(() => {
    if (inputFile.value.length) {
      console.log("ファイル選択イベント取得 !!");
    //   console.log($("#btn-to-submit"));
      $("#btn-to-submit").prop("disabled", false);
    //   alert('Select Fire');
    } else {
      console.log("ファイルキャンセルイベント取得 !!");
    }
    document.body.onfocus = null;
  }, 500);
}