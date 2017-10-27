$(document).ready(function() {
       $('#company').change(function() {

         var company = $('#company').val();
         console.log(company)
         // Make Ajax Request and expect JSON-encoded data
         $.getJSON(
           'http://127.0.0.1:5000/get_models' + '/' + company,
           function(data) {

             // Remove old options
             $('#models').find('option').remove();

             // Add new items
             $.each(data, function(key, val) {
               var option_item = '<option value="' + val + '">' + val + '</option>'
               $('#models').append(option_item);
             });
           }
         );
       });
     });
