$(document).ready(function () {
  $('input[type=radio][name="SwitchCheck"]').change(function () {
    console.log($(this).val());
    reload();
  });
});

function reload() {
  $("#col-lg-6").load(window.location.href + "#col-lg-6");
}
